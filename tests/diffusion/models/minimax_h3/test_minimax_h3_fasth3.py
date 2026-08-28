# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.lora_weights import PackedLoRALayerWeights

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3.fasth3 import (
    FastH3Adapter,
    load_minimax_h3_fasth3_lora,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_RANK = 4
_HIDDEN = 2
_ATTENTION_INNER = 3
_FFN = 5

# (lora_A input dim, lora_B output dim) per checkpoint-spelled target.
_TARGET_DIMS = {
    "attn.to_q": (_HIDDEN, _ATTENTION_INNER),
    "attn.to_k": (_HIDDEN, _ATTENTION_INNER),
    "attn.to_v": (_HIDDEN, _ATTENTION_INNER),
    "attn.to_out.0": (_ATTENTION_INNER, _HIDDEN),
    "ff.net.0.proj": (_HIDDEN, 2 * _FFN),
    "ff.net.2": (_FFN, _HIDDEN),
}


def _request(path, lora_int_id: int = 1) -> LoRARequest:
    return LoRARequest(lora_name="fasth3", lora_int_id=lora_int_id, lora_path=str(path))


def _write_adapter(
    directory,
    *,
    blocks: int = 2,
    refiner_blocks: int = 1,
    config: dict | None = None,
    gate_tensors: bool = False,
    extra_tensors: dict[str, torch.Tensor] | None = None,
    drop_lora_b: str | None = None,
) -> None:
    """Write a FastVideo-style PEFT adapter over the H3 DiT."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for prefix, block_count in (
        ("transformer_blocks", blocks),
        ("token_refiner.refiner_blocks", refiner_blocks),
    ):
        for index in range(block_count):
            for suffix, (input_dim, output_dim) in _TARGET_DIMS.items():
                target = f"{prefix}.{index}.{suffix}"
                tensors[f"{target}.lora_A.weight"] = torch.ones((_RANK, input_dim))
                if target != drop_lora_b:
                    # A distinguishable ramp so the fused-MLP halves stay
                    # identifiable after packing.
                    tensors[f"{target}.lora_B.weight"] = (
                        torch.arange(output_dim, dtype=torch.float32)[:, None].repeat(1, _RANK)
                    )
    if gate_tensors:
        tensors["transformer_blocks.0.attn.to_gate_compress.weight"] = torch.ones((_HIDDEN, _HIDDEN))
    tensors.update(extra_tensors or {})
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    if config is not None:
        (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def _load(directory, *, partition: str = "fl2va", **kwargs):
    return load_minimax_h3_fasth3_lora(
        partition=partition,
        lora_request=_request(directory),
        lora_path=directory,
        dtype=torch.float32,
        **kwargs,
    )


def test_fasth3_adapter_maps_checkpoint_names_onto_the_h3_dit(tmp_path):
    directory = tmp_path / "FastVideo-FastH3-4-step-Preview-v1-LoRA"
    _write_adapter(directory, config={"r": _RANK, "lora_alpha": _RANK})

    lora_model, peft_helper, adapter = _load(directory)

    assert peft_helper.r == _RANK
    assert peft_helper.lora_alpha == _RANK
    assert adapter == FastH3Adapter(base_schedule=None, vsa_gate_skipped=False)
    # Six targets across two DiT blocks and one refiner block.
    assert len(lora_model.loras) == 18
    assert "blocks.1.attn.to_q" in lora_model.loras
    assert "blocks.1.attn.out_proj" in lora_model.loras
    assert "token_refiner.blocks.0.mlp.fc2" in lora_model.loras


def test_fasth3_swaps_the_fused_mlp_halves_into_h3_gate_first_order(tmp_path):
    directory = tmp_path / "fasth3"
    _write_adapter(directory, config={"r": _RANK})

    lora_model, _, _ = _load(directory)

    fc1 = lora_model.loras["blocks.0.mlp.fc1"]
    assert isinstance(fc1, PackedLoRALayerWeights)
    gate_b, up_b = fc1.lora_b
    # The checkpoint stores value-first; H3's fused projection is gate-first, so
    # the second checkpoint half has to come out as the first packed slice.
    assert torch.equal(gate_b[:, 0], torch.arange(_FFN, 2 * _FFN, dtype=torch.float32))
    assert torch.equal(up_b[:, 0], torch.arange(_FFN, dtype=torch.float32))


def test_fasth3_reports_and_skips_the_vsa_compression_gate(tmp_path, caplog):
    directory = tmp_path / "fasth3-vsa"
    _write_adapter(directory, config={"r": _RANK}, gate_tensors=True)

    lora_model, _, adapter = _load(directory)

    assert adapter is not None and adapter.vsa_gate_skipped
    assert not any("gate_compress" in name for name in lora_model.loras)
    assert "Video Sparse Attention gate tensors" in caplog.text


def test_a_gate_payload_identifies_the_release_without_a_recognizable_path(tmp_path):
    named = tmp_path / "checkpoints" / "step_1300"
    _write_adapter(named, config={"r": _RANK}, gate_tensors=True)
    assert _load(named)[2] is not None

    # No gate payload and no release marker: still a loadable H3 adapter, but
    # nothing here says it was distilled, so the four-step contract must not be
    # imposed on it.
    plain = tmp_path / "checkpoints" / "my_style_lora"
    _write_adapter(plain, config={"r": _RANK})
    assert _load(plain)[2] is None


def test_fasth3_reads_a_pinned_rectified_flow_schedule(tmp_path):
    directory = tmp_path / "fasth3"
    _write_adapter(directory, config={"r": _RANK, "_fasth3": {"base_schedule": [1.0, 0.75, 0.5, 0.25, 0.0]}})

    _, _, adapter = _load(directory)

    assert adapter is not None
    assert adapter.base_schedule is not None
    assert adapter.base_schedule.num_inference_steps == 4


def test_fasth3_infers_the_rank_when_the_adapter_ships_no_config(tmp_path):
    directory = tmp_path / "fasth3"
    _write_adapter(directory, config=None)

    _, peft_helper, adapter = _load(directory)

    assert adapter is not None
    assert peft_helper.r == _RANK
    assert peft_helper.lora_alpha == _RANK


def test_fasth3_rejects_ref2va_and_offload_modes(tmp_path):
    directory = tmp_path / "fasth3"
    _write_adapter(directory, config={"r": _RANK})

    with pytest.raises(ValueError, match="distills T2VA only"):
        _load(directory, partition="ref2va")
    with pytest.raises(ValueError, match="does not support layerwise offload"):
        _load(directory, unsupported_offload_mode="layerwise offload")


def test_a_foreign_adapter_falls_through_to_the_generic_peft_loader(tmp_path):
    directory = tmp_path / "wan-lora"
    directory.mkdir()
    save_file(
        {
            "blocks.0.self_attn.q.lora_A.weight": torch.ones((_RANK, _HIDDEN)),
            "blocks.0.self_attn.q.lora_B.weight": torch.ones((_HIDDEN, _RANK)),
        },
        str(directory / "adapter_model.safetensors"),
    )

    assert _load(directory) is None
    assert _load(tmp_path / "missing") is None


def test_fasth3_rejects_a_partially_recognizable_adapter(tmp_path):
    unmapped = tmp_path / "fasth3-unmapped"
    _write_adapter(
        unmapped,
        config={"r": _RANK},
        extra_tensors={"transformer_blocks.0.attn.norm_q.lora_A.weight": torch.ones((_RANK, _HIDDEN))},
    )
    with pytest.raises(ValueError, match="do not name a supported target"):
        _load(unmapped)

    unpaired = tmp_path / "fasth3-unpaired"
    _write_adapter(unpaired, config={"r": _RANK}, drop_lora_b="transformer_blocks.0.attn.to_q")
    with pytest.raises(ValueError, match="unpaired A/B tensors"):
        _load(unpaired)


def test_a_snapshot_holding_several_variants_is_not_guessed_at(tmp_path):
    directory = tmp_path / "FastH3-preview"
    directory.mkdir()
    for name in ("vsa_datafree.safetensors", "dense_datafree.safetensors"):
        save_file({"transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones((_RANK, _HIDDEN))}, str(directory / name))

    assert _load(directory) is None


def test_the_legacy_manager_loads_fasth3_without_changing_its_interface(tmp_path):
    directory = tmp_path / "fasth3"
    _write_adapter(directory, config={"r": _RANK})

    class _Pipeline:
        def _load_diffusion_lora_adapter(self, **kwargs):
            lora_model, peft_helper, _ = load_minimax_h3_fasth3_lora(partition="fl2va", **kwargs)
            return lora_model, peft_helper

    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = _Pipeline()
    manager.dtype = torch.float32
    manager._expected_lora_modules = {"to_q", "fc1"}

    lora_model, peft_helper = manager._load_adapter(_request(directory))

    assert lora_model.id == 1
    assert peft_helper.r == _RANK
    assert "blocks.0.attn.to_q" in lora_model.loras


def _pipeline_stub():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline._turbo_lora_adapter_ids = set()
    pipeline._fasth3_lora_adapters = {}
    return pipeline


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_an_active_fasth3_adapter_restricts_requests_to_t2va(task):
    pipeline = _pipeline_stub()
    pipeline._fasth3_lora_adapters[1] = FastH3Adapter(base_schedule=None, vsa_gate_skipped=False)
    sampling = SimpleNamespace(lora_request=_request("fasth3"), lora_scale=1.0)

    assert pipeline._active_fasth3_adapter(sampling) is not None
    with pytest.raises(OmniClientError, match="distills \\['t2va'\\] only"):
        pipeline._resolve_task(task, {}, has_fasth3_lora=True)
    assert pipeline._resolve_task(task, {}, has_fasth3_lora=False) == task

    # A deactivated adapter constrains nothing.
    assert pipeline._active_fasth3_adapter(SimpleNamespace(lora_request=sampling.lora_request, lora_scale=0.0)) is None


@pytest.mark.parametrize("num_inference_steps", [None, 4, 50, "5"])
def test_fasth3_requires_five_sigma_points_for_four_denoiser_calls(num_inference_steps):
    pipeline = _pipeline_stub()
    adapter = FastH3Adapter(base_schedule=None, vsa_gate_skipped=False)

    with pytest.raises(OmniClientError, match="num_inference_steps=5"):
        pipeline._validate_fasth3_sampling(SimpleNamespace(num_inference_steps=num_inference_steps), adapter)
    pipeline._validate_fasth3_sampling(SimpleNamespace(num_inference_steps=5), adapter)


def test_a_pinned_schedule_owns_the_step_count_instead():
    from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

    pipeline = _pipeline_stub()
    adapter = FastH3Adapter(
        base_schedule=DMD2SigmaSchedule.from_positions([1.0, 0.75, 0.5, 0.25, 0.0]),
        vsa_gate_skipped=False,
    )

    # The pinned schedule is enforced in interval terms where sigmas are built,
    # so the sigma-point check must not fire a second, contradictory rule here.
    pipeline._validate_fasth3_sampling(SimpleNamespace(num_inference_steps=4), adapter)


def test_reloading_an_adapter_id_replaces_its_release_classification(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    pipeline = _pipeline_stub()
    pipeline.od_config = SimpleNamespace(enable_cpu_offload=False, enable_layerwise_offload=False)
    request = _request("fasth3")

    def load():
        return pipeline._load_diffusion_lora_adapter(
            lora_request=request,
            lora_path=request.lora_path,
            dtype=torch.float32,
        )

    lora_model, peft_helper = object(), object()
    adapter = FastH3Adapter(base_schedule=None, vsa_gate_skipped=False)
    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", lambda **_: None)
    monkeypatch.setattr(
        pipeline_module,
        "load_minimax_h3_fasth3_lora",
        lambda **_: (lora_model, peft_helper, adapter),
    )
    assert load() == (lora_model, peft_helper)
    assert pipeline._fasth3_lora_adapters == {1: adapter}

    # A manager eviction can be followed by a different adapter reusing the ID.
    monkeypatch.setattr(pipeline_module, "load_minimax_h3_fasth3_lora", lambda **_: None)
    assert load() is None
    assert pipeline._fasth3_lora_adapters == {}


def test_a_turbo_artifact_still_wins_the_dispatch(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    pipeline = _pipeline_stub()
    pipeline.od_config = SimpleNamespace(enable_cpu_offload=False, enable_layerwise_offload=False)
    turbo = (object(), object())
    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", lambda **_: turbo)
    monkeypatch.setattr(
        pipeline_module,
        "load_minimax_h3_fasth3_lora",
        lambda **_: pytest.fail("the FastH3 loader must not see a recognized Turbo artifact"),
    )

    loaded = pipeline._load_diffusion_lora_adapter(
        lora_request=_request("turbo"),
        lora_path="turbo",
        dtype=torch.float32,
    )

    assert loaded is turbo
    assert pipeline._turbo_lora_adapter_ids == {1}
    assert pipeline._fasth3_lora_adapters == {}


def test_binding_validation_names_the_incomplete_release():
    pipeline = _pipeline_stub()
    pipeline._fasth3_lora_adapters[1] = FastH3Adapter(base_schedule=None, vsa_gate_skipped=False)
    lora_model = SimpleNamespace(id=1, loras={"to_q": object(), "to_k": object()})

    pipeline._validate_diffusion_lora_binding(
        lora_model=lora_model,
        bound_lora_names=frozenset(lora_model.loras),
    )
    with pytest.raises(ValueError, match="FastH3 LoRA binding is incomplete: bound=1/2"):
        pipeline._validate_diffusion_lora_binding(
            lora_model=lora_model,
            bound_lora_names=frozenset({"to_q"}),
        )


def test_fasth3_carries_rslora_scaling_and_refuses_full_weight_modules(tmp_path):
    plain = tmp_path / "fasth3-plain"
    _write_adapter(plain, config={"r": _RANK, "lora_alpha": 8})
    rslora = tmp_path / "fasth3-rslora"
    _write_adapter(rslora, config={"r": _RANK, "lora_alpha": 8, "use_rslora": True})

    # rsLoRA scales by alpha/sqrt(r) instead of alpha/r, so the flag has to
    # survive the load or the adapter is applied at the wrong strength.
    assert _load(plain)[0].loras["blocks.0.attn.to_q"].scaling == pytest.approx(8 / _RANK)
    assert _load(rslora)[0].loras["blocks.0.attn.to_q"].scaling == pytest.approx(8 / _RANK**0.5)

    saved = tmp_path / "fasth3-modules-to-save"
    _write_adapter(saved, config={"r": _RANK, "modules_to_save": ["proj_out"]})
    with pytest.raises(ValueError, match="unsupported modules_to_save"):
        _load(saved)
