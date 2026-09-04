# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loader for the LightX2V Turbo MiniMax-H3 LoRA family.

LightX2V publishes a matrix of Turbo artifacts rather than a single file:
``fl2v``/``ref2v`` task families, four- and eight-step distillations, and 544p
and 768p training resolutions.  They share rank 128 and target set but differ
in LoRA alpha and in the sampler contract they were distilled for (sigma points
and flow shift), so :class:`TurboSpec` carries what a given file needs and the
pipeline validates a request against the artifact actually loaded.

Only the Diffusers-PEFT exports are served.  LightX2V also ships a ComfyUI
export of most artifacts; those fuse Q/K/V into one projection, so this loader
refuses them by name instead of guessing at the layout.

The native FlashGen contract lives in ``.npu.lora``; that artifact differs in
rank, target naming and QKV layout, so it owns its own parsing and packing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.lora.request import LoRARequest

logger = init_logger(__name__)

_TURBO_RANK = 128
_TURBO_HIDDEN_SIZE = 5376
_TURBO_ATTENTION_INNER_SIZE = 7168
_TURBO_FFN_HIDDEN_SIZE = 14336
# Every published Diffusers-layout Turbo artifact is named
# ``minimax_h3_<task>_turbo_<n>step_v<major.minor>[_768p][_bf16]``.
# The name is the only place the sampler contract is recorded, so it -- not a
# single hard-coded filename -- decides which requests the adapter accepts.
_TURBO_NAME_RE = re.compile(
    r"^minimax_h3_(?P<task>fl2v|ref2v)_turbo_(?P<steps>\d+)step"
    r"_v(?P<version>\d+\.\d+)(?P<res>_768p)?(?:_bf16)?\.safetensors$"
)
# The ComfyUI exports share that stem but carry a fused-QKV layout. They are
# named so a mistaken download is refused with the reason rather than falling
# through to a loader that cannot read them either.
_COMFYUI_MARKER = "_comfyui"
# The 768p retrains moved to a shorter video flow shift; the original
# mixed-aspect 544p artifacts keep the base model's shift.  Audio shift is 3.0
# across the family.
_TURBO_VIDEO_SHIFT_768P = 6.0
_TURBO_VIDEO_SHIFT_544P = 12.0
_TURBO_AUDIO_SHIFT = 3.0
# ``minimax_h3_fl2v_turbo_4step_v0.1.safetensors`` is the one artifact that
# declares no alpha.  Both later four-step FL2VA artifacts declare alpha ==
# rank, so scale 1.0 is the documented assumption; request-level ``scale``
# remains available to override it.
_TURBO_DEFAULT_ALPHA = float(_TURBO_RANK)


@dataclass(frozen=True)
class TurboSpec:
    """The sampler contract and tensor layout of one Turbo artifact."""

    filename: str
    task_family: str
    """``fl2v`` (serves t2va and fl2va) or ``ref2v`` (serves ref2va)."""
    version: str
    denoise_steps: int
    video_shift: float
    audio_shift: float
    rank: int
    alpha: float

    @property
    def sigma_points(self) -> int:
        """Sigma points the API contract expects: one more than the forwards."""
        return self.denoise_steps + 1

    @property
    def supported_tasks(self) -> frozenset[str]:
        if self.task_family == "ref2v":
            return frozenset({"ref2va"})
        return frozenset({"t2va", "fl2va"})


def parse_turbo_filename(name: str) -> dict[str, object] | None:
    """Return the contract encoded in a Turbo filename, or ``None``."""

    match = _TURBO_NAME_RE.match(name)
    if match is None:
        return None
    steps = int(match.group("steps"))
    if steps <= 0:
        return None
    return {
        "task_family": match.group("task"),
        "version": match.group("version"),
        "denoise_steps": steps,
        "video_shift": (_TURBO_VIDEO_SHIFT_768P if match.group("res") else _TURBO_VIDEO_SHIFT_544P),
        "audio_shift": _TURBO_AUDIO_SHIFT,
    }


_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"
_TURBO_TARGETS = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"})
_TURBO_RAW_TARGET_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_TURBO_TARGET_DIMS = {
    "attn.to_q": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_k": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_v": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_out.0": (_TURBO_ATTENTION_INNER_SIZE, _TURBO_HIDDEN_SIZE),
    "ff.net.0.proj": (_TURBO_HIDDEN_SIZE, 2 * _TURBO_FFN_HIDDEN_SIZE),
    "ff.net.2": (_TURBO_FFN_HIDDEN_SIZE, _TURBO_HIDDEN_SIZE),
}
_TURBO_EXPECTED_RAW_TARGETS = frozenset(
    f"{prefix}.{block_index}.{suffix}"
    for prefix, block_count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    )
    for block_index in range(block_count)
    for suffix in _TURBO_RAW_TARGET_SUFFIXES
)
_TURBO_TARGET_PATTERN = (
    r"^transformer\.(?:token_refiner\.blocks|blocks)\.\d+\."
    r"(?:attn\.(?:to_q|to_k|to_v|out_proj)|mlp\.(?:fc1|fc2))$"
)

_TURBO_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "token_refiner.refiner_blocks.": "token_refiner.blocks.",
        "transformer_blocks.": "blocks.",
        ".attn.to_out.0.": ".attn.out_proj.",
        ".ff.net.0.proj.": ".mlp.fc1.",
        ".ff.net.2.": ".mlp.fc2.",
        ".lora_A.default.": ".lora_A.",
        ".lora_B.default.": ".lora_B.",
    }
)


def _select_turbo_file(artifact_path: str | Path) -> Path | None:
    path = Path(artifact_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None

    # A directory may hold several artifacts of the family; only an
    # unambiguous single candidate can be selected without a request-level
    # choice, so anything else is left to the caller to name explicitly.
    candidates = sorted(child for child in path.glob("*.safetensors") if parse_turbo_filename(child.name))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{path} holds {len(candidates)} MiniMax-H3 Turbo artifacts "
            f"({[c.name for c in candidates[:4]]}); point --lora-path at one file."
        )
    return None


def _validate_and_convert_tensors(checkpoint) -> dict[str, torch.Tensor]:
    """Validate a Diffusers-layout Turbo tensor set and pack it for the manager."""

    tensors: dict[str, torch.Tensor] = {}
    pairs: dict[str, set[str]] = {}
    raw_targets: set[str] = set()
    for name in checkpoint.keys():
        if name.endswith(_LORA_A_SUFFIX):
            raw_target = name[: -len(_LORA_A_SUFFIX)]
            side = "a"
        elif name.endswith(_LORA_B_SUFFIX):
            raw_target = name[: -len(_LORA_B_SUFFIX)]
            side = "b"
        else:
            raise ValueError(f"Unconsumed MiniMax-H3 Turbo tensor: {name!r}")
        raw_targets.add(raw_target)

        mapped_name = _TURBO_WEIGHTS_MAPPER.apply_list([name])[0]
        mapped_target = mapped_name.rsplit(".lora_", 1)[0]
        if mapped_target.rsplit(".", 1)[-1] not in _TURBO_TARGETS:
            raise ValueError(f"Unsupported MiniMax-H3 Turbo target: {raw_target!r}")
        target_sides = pairs.setdefault(mapped_target, set())
        if side in target_sides:
            raise ValueError(f"Duplicate MiniMax-H3 Turbo tensor for {mapped_target}.{side}")
        target_sides.add(side)

        tensor = checkpoint.get_tensor(name)
        if tensor.ndim != 2:
            raise ValueError(f"MiniMax-H3 Turbo LoRA tensors must be matrices, got {name}={tuple(tensor.shape)}")
        suffix = next((suffix for suffix in _TURBO_RAW_TARGET_SUFFIXES if raw_target.endswith(suffix)), None)
        if suffix is None:
            raise ValueError(f"MiniMax-H3 Turbo LoRA contains unsupported target: {raw_target}")
        input_dim, output_dim = _TURBO_TARGET_DIMS[suffix]
        expected_shape = (_TURBO_RANK, input_dim) if side == "a" else (output_dim, _TURBO_RANK)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MiniMax-H3 Turbo tensor has invalid global shape: {name}={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )
        if side == "b" and ".ff.net.0.proj." in name:
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0).contiguous()
        tensors[name] = tensor

    incomplete = sorted(target for target, sides in pairs.items() if sides != {"a", "b"})
    if incomplete:
        raise ValueError(f"Incomplete MiniMax-H3 Turbo LoRA pairs: {incomplete}")
    missing = sorted(_TURBO_EXPECTED_RAW_TARGETS - raw_targets)
    unexpected = sorted(raw_targets - _TURBO_EXPECTED_RAW_TARGETS)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 Turbo target set does not match the published artifact layout: "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )
    return tensors


def _pack_h3_turbo_fc1(lora_model: LoRAModel) -> None:
    """Represent H3's fused gate/up projection without generic layout guesses."""

    for module_name, weights in tuple(lora_model.loras.items()):
        if not module_name.endswith(".mlp.fc1"):
            continue
        gate_b, up_b = weights.lora_b.chunk(2, dim=0)
        lora_model.loras[module_name] = PackedLoRALayerWeights(
            module_name=module_name,
            rank=weights.rank,
            lora_alphas=[weights.lora_alpha, weights.lora_alpha],
            lora_a=[weights.lora_a, weights.lora_a],
            lora_b=[gate_b.contiguous(), up_b.contiguous()],
            scaling=[weights.scaling, weights.scaling],
        )


def load_minimax_h3_turbo_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper, TurboSpec] | None:
    """Load any published LightX2V Turbo artifact through the legacy manager."""

    lora_file = _select_turbo_file(lora_path)
    if lora_file is None:
        return None
    fields = parse_turbo_filename(lora_file.name)
    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if fields is None:
            if lora_file.name.startswith("minimax_h3_") and _COMFYUI_MARKER in lora_file.name:
                raise ValueError(
                    f"{lora_file.name} is a ComfyUI-layout MiniMax-H3 Turbo export, which is not "
                    "supported; download the Diffusers export of the same artifact instead."
                )
            # Not a Turbo artifact by name.  Only claim it when the metadata
            # says it is one, so other adapters fall through to their loader.
            if metadata.get("key_format") != "minimax-h3-diffusers":
                return None
            raise ValueError(
                f"{lora_file.name!r} carries MiniMax-H3 Turbo metadata but its name does not follow "
                "minimax_h3_<fl2v|ref2v>_turbo_<n>step_v<x.y>[_768p][_bf16].safetensors, "
                "so its sampler contract cannot be determined."
            )

        # Three published artifacts (both v0.1 files and the 544p eight-step
        # one) declare no key_format at all, so absence is not an error; a
        # declaration that names a *different* format is.
        declared_format = metadata.get("key_format")
        if declared_format is not None and declared_format != "minimax-h3-diffusers":
            raise ValueError(
                f"{lora_file.name} is named as a MiniMax-H3 Turbo artifact but requires safetensors "
                f"metadata key_format='minimax-h3-diffusers', got {declared_format!r}"
            )
        raw_alpha = metadata.get("alpha")
        if raw_alpha is None:
            alpha = _TURBO_DEFAULT_ALPHA
            logger.warning(
                "MiniMax-H3 Turbo artifact %s declares no alpha; assuming alpha=%g (scale 1.0). "
                "Override with the request-level LoRA scale if output looks over- or under-driven.",
                lora_file.name,
                alpha,
            )
        else:
            try:
                alpha = float(raw_alpha)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"MiniMax-H3 Turbo alpha must be numeric, got {raw_alpha!r}") from exc
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"MiniMax-H3 Turbo alpha must be a positive number, got {raw_alpha!r}")

        spec = TurboSpec(
            filename=lora_file.name,
            task_family=str(fields["task_family"]),
            version=str(fields["version"]),
            denoise_steps=int(fields["denoise_steps"]),
            video_shift=float(fields["video_shift"]),
            audio_shift=float(fields["audio_shift"]),
            rank=_TURBO_RANK,
            alpha=alpha,
        )
        if partition == "ref2va" and spec.task_family == "fl2v":
            raise ValueError(f"{spec.filename} is an FL2VA/T2VA Turbo artifact; a Ref2VA-only server cannot serve it.")
        # A ``combined`` server builds the Ref2VA DiT as ``transformers_ref``
        # and serves ref2va requests from it, but the LoRA target pattern only
        # injects into ``transformer`` -- the FL2VA DiT. A Ref2VA adapter loaded
        # there binds to the stack that never runs, leaving ref2va requests to
        # execute an undistilled DiT on the artifact's few-step schedule. Only a
        # Ref2VA-only server, whose single DiT *is* ``transformer``, can serve it.
        if spec.task_family == "ref2v" and partition != "ref2va":
            raise ValueError(
                f"{spec.filename} is a Ref2VA Turbo artifact; start the server with --task-type ref2va "
                f"(task_type={partition!r} serves ref2va from a DiT the adapter cannot bind to)"
            )
        if unsupported_offload_mode is not None:
            raise ValueError(f"MiniMax-H3 Turbo dynamic LoRA does not support {unsupported_offload_mode}")

        tensors = _validate_and_convert_tensors(checkpoint)

    peft_helper = PEFTHelper.from_dict(
        {
            "r": spec.rank,
            "lora_alpha": spec.alpha,
            "target_modules": _TURBO_TARGET_PATTERN,
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=_TURBO_WEIGHTS_MAPPER,
    )
    _pack_h3_turbo_fc1(lora_model)
    logger.info(
        "MiniMax-H3 Turbo adapter %s: task=%s v%s, %d denoiser forwards (%d sigma points), "
        "rank=%d alpha=%g, flow_shift=%g/%g",
        spec.filename,
        spec.task_family,
        spec.version,
        spec.denoise_steps,
        spec.sigma_points,
        spec.rank,
        spec.alpha,
        spec.video_shift,
        spec.audio_shift,
    )
    return lora_model, peft_helper, spec
