# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.fasth3 import (
    FASTH3_FLOW_SHIFT,
    FASTH3_FORMAT,
    FASTH3_SIGMA_POINTS,
    FastH3AdapterError,
    FastH3WeightFusion,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _reorder_grouped_qkv_to_qkv
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_RANK = 2
_HIDDEN = 4
_HEAD_DIM = 2
_HEADS = 3
_INNER = _HEAD_DIM * _HEADS  # attention inner size
_FFN = 5


def _factors(out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A rank-2 pair whose product is reproducible and non-symmetric."""
    a = torch.arange(_RANK * in_dim, dtype=torch.float32).reshape(_RANK, in_dim) / (in_dim * _RANK)
    b = torch.arange(out_dim * _RANK, dtype=torch.float32).reshape(out_dim, _RANK) / (out_dim * _RANK)
    return a, b


def _write_adapter(path, *, tensors=None, drop: str | None = None) -> None:
    """Write a single-block artifact in the published ``fastvideo-lora-v2`` shape."""
    payload: dict[str, torch.Tensor] = {}
    for suffix, (out_dim, in_dim) in {
        "attn.to_q": (_INNER, _HIDDEN),
        "attn.to_k": (_INNER, _HIDDEN),
        "attn.to_v": (_INNER, _HIDDEN),
        "attn.to_out.0": (_HIDDEN, _INNER),
        "ff.net.0.proj": (2 * _FFN, _HIDDEN),
        "ff.net.2": (_HIDDEN, _FFN),
    }.items():
        a, b = _factors(out_dim, in_dim)
        payload[f"transformer_blocks.0.{suffix}.lora_A.weight"] = a
        payload[f"transformer_blocks.0.{suffix}.lora_B.weight"] = b
    payload.update(tensors or {})
    if drop is not None:
        payload.pop(drop, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(path), metadata={"format": FASTH3_FORMAT, "rank": str(_RANK)})


def _claim(path, **kwargs) -> FastH3WeightFusion | None:
    """Whatever from_path decides, including declining the artifact."""
    return FastH3WeightFusion.from_path(path, head_dim=kwargs.pop("head_dim", _HEAD_DIM), **kwargs)


def _load(path, **kwargs) -> FastH3WeightFusion:
    """The same, for the tests that only make sense on a claimed adapter."""
    fusion = _claim(path, **kwargs)
    assert fusion is not None, f"{path} was not claimed as a FastH3 adapter"
    return fusion


def test_only_a_fastvideo_lora_v2_artifact_is_claimed(tmp_path):
    plain = tmp_path / "peft" / "adapter_model.safetensors"
    plain.parent.mkdir(parents=True)
    save_file({"transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones((_RANK, _HIDDEN))}, str(plain))
    # No fastvideo-lora-v2 metadata: this is somebody else's LoRA and has to
    # stay on the dynamic route rather than being fused.
    assert _claim(plain.parent) is None
    assert _claim(tmp_path / "missing") is None

    claimed = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(claimed)
    assert _claim(claimed.parent) is not None


def test_the_published_bundle_root_is_refused_rather_than_guessed(tmp_path):
    root = tmp_path / "FastVideo-FastH3-4-step-Preview-v1-LoRA"
    for slug in ("dense-datafree", "vsa-datafree"):
        _write_adapter(root / slug / "adapter_model.safetensors")
    (root / "adapter_manifest.json").write_text(json.dumps({"schema_version": "fasth3-lora-bundle-v1"}))

    with pytest.raises(FastH3AdapterError, match="point --lora-path at one variant"):
        _load(root)
    # One variant inside it is unambiguous.
    assert _claim(root / "dense-datafree") is not None


def test_low_rank_factors_reach_the_fused_projections(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    fusion = _load(path.parent)

    qkv = torch.zeros((3 * _INNER, _HIDDEN), dtype=torch.float32)
    fused = fusion.fuse("blocks.0.attn.qkv_proj.weight", qkv).cpu()
    # The checkpoint stores one head group at a time as [q, k, v], so the delta
    # has to survive the loader's own unpacking as three separate projections.
    q, k, v = torch.split(
        _reorder_grouped_qkv_to_qkv(fused, num_query_groups=_HEADS, heads_per_group=1, head_dim=_HEAD_DIM),
        [_INNER, _INNER, _INNER],
    )
    a, b = _factors(_INNER, _HIDDEN)
    expected = b @ a
    for got in (q, k, v):
        assert torch.allclose(got, expected, atol=1e-5)


def test_the_fused_mlp_delta_is_swapped_into_gate_first_order(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    fusion = _load(path.parent)

    fused = fusion.fuse("blocks.0.mlp.fc1.weight", torch.zeros((2 * _FFN, _HIDDEN))).cpu()
    a, b = _factors(2 * _FFN, _HIDDEN)
    value, gate = (b @ a).chunk(2, dim=0)
    # The diffusers export is value-first; H3's fc1 is gate-first.
    assert torch.allclose(fused, torch.cat((gate, value), dim=0), atol=1e-5)


def test_diff_and_diff_b_edit_weights_and_biases(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(
        path,
        tensors={
            "transformer_blocks.0.norm1.diff": torch.full((_HIDDEN,), 0.25),
            "transformer_blocks.0.adaln_proj.linear.diff_b": torch.full((_HIDDEN,), -0.5),
            "context_embedder.diff": torch.full((_HIDDEN, _HIDDEN), 0.125),
            "context_embedder.diff_b": torch.full((_HIDDEN,), 2.0),
        },
    )
    fusion = _load(path.parent)

    # A full-rank delta on an RMSNorm vector is exactly what a LoRA layer
    # cannot express, and why this release is fused instead of switched.
    assert torch.allclose(fusion.fuse("blocks.0.norm1.weight", torch.ones(_HIDDEN)).cpu(), torch.full((_HIDDEN,), 1.25))
    assert torch.allclose(
        fusion.fuse("blocks.0.adaln_proj.linear.bias", torch.ones(_HIDDEN)).cpu(), torch.full((_HIDDEN,), 0.5)
    )
    assert torch.allclose(
        fusion.fuse("condition_proj.weight", torch.zeros((_HIDDEN, _HIDDEN))).cpu(),
        torch.full((_HIDDEN, _HIDDEN), 0.125),
    )
    assert torch.allclose(fusion.fuse("condition_proj.bias", torch.zeros(_HIDDEN)).cpu(), torch.full((_HIDDEN,), 2.0))


def test_an_unpatched_parameter_passes_through_untouched(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    fusion = _load(path.parent)

    weight = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    assert fusion.fuse("blocks.0.attn.q_norm.weight", weight) is weight


def test_the_fused_result_keeps_the_parameter_dtype(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, tensors={"transformer_blocks.0.norm1.diff": torch.full((_HIDDEN,), 0.5)})
    fusion = _load(path.parent)

    fused = fusion.fuse("blocks.0.norm1.weight", torch.ones(_HIDDEN, dtype=torch.bfloat16))
    assert fused.dtype == torch.bfloat16


def test_a_vsa_variant_is_recognised_by_its_compression_gates(tmp_path):
    dense = tmp_path / "dense" / "adapter_model.safetensors"
    _write_adapter(dense)
    assert _load(dense.parent).requires_vsa is False

    sparse = tmp_path / "vsa" / "adapter_model.safetensors"
    _write_adapter(sparse, tensors={"transformer_blocks.0.attn.to_gate_compress.set_weight": torch.ones((2, 2))})
    assert _load(sparse.parent).requires_vsa is True


def test_a_tensor_naming_no_h3_parameter_is_an_error(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, tensors={"transformer_blocks.0.attn.norm_q.lora_A.weight": torch.ones((_RANK, _HIDDEN))})
    # Dropping it silently would load a model that is not the distilled student.
    with pytest.raises(FastH3AdapterError, match="name no known"):
        _load(path.parent)


def test_an_unpaired_low_rank_factor_is_an_error(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, drop="transformer_blocks.0.attn.to_q.lora_B.weight")

    with pytest.raises(FastH3AdapterError, match="unpaired factor"):
        _load(path.parent)


def test_a_delta_the_checkpoint_never_offered_is_reported(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    fusion = _load(path.parent)

    fusion.fuse("blocks.0.attn.qkv_proj.weight", torch.zeros((3 * _INNER, _HIDDEN)))
    # An unapplied delta is the failure that matters: the model would load and
    # generate, just not as the distilled student.
    with pytest.raises(FastH3AdapterError, match="never provided"):
        fusion.validate_fully_applied()

    for name, shape in (
        ("blocks.0.attn.out_proj.weight", (_HIDDEN, _INNER)),
        ("blocks.0.mlp.fc1.weight", (2 * _FFN, _HIDDEN)),
        ("blocks.0.mlp.fc2.weight", (_HIDDEN, _FFN)),
    ):
        fusion.fuse(name, torch.zeros(shape))
    fusion.validate_fully_applied()


def test_apply_fuses_a_whole_weight_stream(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, tensors={"transformer_blocks.0.norm1.diff": torch.full((_HIDDEN,), 3.0)})
    fusion = _load(path.parent)

    streamed = dict(
        fusion.apply(
            [
                ("blocks.0.norm1.weight", torch.zeros(_HIDDEN)),
                ("blocks.0.attn.q_norm.weight", torch.zeros(_HIDDEN)),
            ]
        )
    )
    assert torch.allclose(streamed["blocks.0.norm1.weight"].cpu(), torch.full((_HIDDEN,), 3.0))
    assert torch.allclose(streamed["blocks.0.attn.q_norm.weight"].cpu(), torch.zeros(_HIDDEN))


def _pipeline_stub(fusion=None):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline._turbo_lora_adapter_ids = set()
    pipeline._fasth3 = fusion
    return pipeline


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_an_active_fasth3_fusion_restricts_requests_to_t2va(task, tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    pipeline = _pipeline_stub(_load(path.parent))

    with pytest.raises(OmniClientError, match="distills \\['t2va'\\] only"):
        pipeline._resolve_task(task, {})
    assert pipeline._resolve_task("t2va", {}) == "t2va"
    assert _pipeline_stub()._resolve_task(task, {}) == task


@pytest.mark.parametrize("num_inference_steps", [None, 4, 50, "5"])
def test_fasth3_requires_five_sigma_points_for_four_forwards(num_inference_steps, tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    pipeline = _pipeline_stub(_load(path.parent))

    with pytest.raises(OmniClientError, match=f"num_inference_steps={FASTH3_SIGMA_POINTS}"):
        pipeline._validate_fasth3_sampling(SimpleNamespace(num_inference_steps=num_inference_steps))
    pipeline._validate_fasth3_sampling(SimpleNamespace(num_inference_steps=FASTH3_SIGMA_POINTS))


def test_adopting_the_contract_neutralises_the_rectified_flow_shift(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    pipeline = _pipeline_stub(_load(path.parent))
    pipeline.partition = "fl2va"
    pipeline.default_video_shift, pipeline.default_audio_shift = 12.0, 3.0

    pipeline._adopt_fasth3_contract()

    # The student's four-jump ladder is uniform, so base H3's shifts would move
    # every sampling point off the rungs it was trained on.
    assert pipeline.default_video_shift == FASTH3_FLOW_SHIFT
    assert pipeline.default_audio_shift == FASTH3_FLOW_SHIFT


def test_adopting_the_contract_refuses_a_vsa_variant_and_ref2va(tmp_path):
    sparse = tmp_path / "vsa" / "adapter_model.safetensors"
    _write_adapter(sparse, tensors={"transformer_blocks.0.attn.to_gate_compress.set_weight": torch.ones((2, 2))})
    pipeline = _pipeline_stub(_load(sparse.parent))
    pipeline.partition = "fl2va"
    with pytest.raises(ValueError, match="Video Sparse Attention variant"):
        pipeline._adopt_fasth3_contract()

    dense = tmp_path / "dense" / "adapter_model.safetensors"
    _write_adapter(dense)
    pipeline = _pipeline_stub(_load(dense.parent))
    pipeline.partition = "ref2va"
    with pytest.raises(ValueError, match="cannot serve a Ref2VA partition"):
        pipeline._adopt_fasth3_contract()


def test_a_full_rank_delta_on_a_fused_parameter_is_refused(tmp_path):
    # Only low-rank factors are placed into H3's fused QKV and gate/up layouts,
    # so a .diff aimed at one would otherwise be added transposed.
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, tensors={"transformer_blocks.0.attn.to_q.diff": torch.ones((_INNER, _HIDDEN))})
    with pytest.raises(FastH3AdapterError, match="fused layout"):
        _claim(path.parent)


def test_a_pipeline_that_fused_its_adapter_needs_no_lora_manager(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)

    assert _pipeline_stub(_load(path.parent)).lora_is_fused is True
    assert _pipeline_stub().lora_is_fused is False
