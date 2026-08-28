# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FastVideo FastH3: a four-step DMD2-distilled LoRA over MiniMax-H3.

FastH3 is a student of the 33B H3 transformer trained with Distribution
Matching Distillation (DMD2). It replaces H3's 49 denoiser evaluations with
four, and ships as a LoRA over the base checkpoint rather than as a full
release, so it reuses H3's text encoder, video VAE, audio VAE, tokenizers and
schedulers unchanged.

Two things separate it from the LightX2V Turbo artifact handled in
:mod:`vllm_omni.diffusion.models.minimax_h3.lora`:

* Turbo is a single published file with a pinned contract (one filename, one
  rank, one exhaustive target set), so its loader validates that contract
  exactly. FastH3 is a preview line with several checkpoint variants, so this
  loader reads the layout from ``adapter_config.json`` and accepts any subset
  of the supported targets.
* Some FastH3 variants are trained with Video Sparse Attention and carry a
  ``to_gate_compress`` compression-gate payload alongside the LoRA tensors.
  vLLM-Omni does not yet run VSA over H3's packed ``[text | cond | audio |
  video]`` sequence, so that payload is reported and skipped: the adapter runs
  dense, which is the mode FastVideo exposes as ``--no-vsa``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import torch
from safetensors import safe_open
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.lora.request import LoRARequest

from .lora import pack_minimax_h3_fc1

logger = init_logger(__name__)

# Five sigma boundaries bound four intervals, i.e. the four DiT calls the
# student was distilled for.
FASTH3_SIGMA_POINTS = 5
# Preview v1 distills the text-to-video-and-audio path only; FL2VA and Ref2VA
# students are announced but not released.
FASTH3_SUPPORTED_TASKS = frozenset({"t2va"})

_CONFIG_FILENAME = "adapter_config.json"
_GATE_MARKER = "to_gate_compress"
_LORA_A_MARKER = ".lora_A"
_LORA_B_MARKER = ".lora_B"

# Prefixes PEFT and the diffusers/FastVideo exporters put in front of the
# transformer path. Stripped left to right until none applies.
_STRIPPED_PREFIXES = (
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
)

# Checkpoint spelling -> vLLM-Omni MiniMaxH3DiTModel spelling. Ordered: the
# refiner rewrite has to run before the generic ``transformer_blocks.`` one.
_NAME_REWRITES = (
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("transformer_blocks.", "blocks."),
    (".attn.to_out.0.", ".attn.out_proj."),
    (".attn.to_out.", ".attn.out_proj."),
    (".ff.net.0.proj.", ".mlp.fc1."),
    (".ff.net.2.", ".mlp.fc2."),
)

# Suffixes that name a linear vLLM-Omni exposes to LoRA on the H3 DiT. ``to_q``
# / ``to_k`` / ``to_v`` bind into the fused ``qkv_proj`` and ``fc1`` into the
# fused gate/up projection; the manager resolves both from the model's
# stacked_params_mapping.
_SUPPORTED_TARGETS = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"})

# Matched against full module names, which carry the component prefix.
_TARGET_PATTERN = (
    r"^transformer\.(?:token_refiner\.blocks|blocks)\.\d+\."
    r"(?:attn\.(?:to_q|to_k|to_v|out_proj)|mlp\.(?:fc1|fc2))$"
)
_MAPPED_TARGET_RE = re.compile(
    r"^(?:token_refiner\.blocks|blocks)\.\d+\.(?:attn\.(?:to_q|to_k|to_v|out_proj)|mlp\.(?:fc1|fc2))$"
)


@dataclass(frozen=True)
class FastH3Adapter:
    """What a loaded FastH3 adapter pins for the requests that use it."""

    # Present only when the release ships explicit rectified-flow positions.
    # Without one the four steps come from the uniform five-point schedule,
    # which is what the published preview configs use.
    base_schedule: DMD2SigmaSchedule | None
    # A VSA-trained variant whose compression gate was skipped, so it is
    # running dense. Kept so the pipeline can say so once per request cycle.
    vsa_gate_skipped: bool


def _resolve_adapter_files(lora_path: str | Path) -> tuple[Path, Path | None] | None:
    """Return ``(weights, config)`` for a PEFT-style adapter, or None."""
    path = Path(lora_path)
    if path.is_file():
        return (path, None) if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None

    config = path / _CONFIG_FILENAME
    named = path / "adapter_model.safetensors"
    if named.is_file():
        return named, config if config.is_file() else None

    candidates = sorted(path.glob("*.safetensors"))
    if len(candidates) != 1:
        # Zero candidates is not an adapter; several is a repository snapshot
        # holding more than one variant, and picking one of them here would be
        # a guess about which checkpoint the user meant.
        return None
    return candidates[0], config if config.is_file() else None


def _read_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"MiniMax-H3 LoRA has an unreadable {_CONFIG_FILENAME}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"MiniMax-H3 LoRA {_CONFIG_FILENAME} must hold an object")
    return config


def _map_module_name(raw_target: str) -> str | None:
    """Rewrite a checkpoint target path to its DiT module name, or None."""
    name = raw_target
    for prefix in _STRIPPED_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    # The rewrites are written with dots on both sides so they cannot match a
    # partial identifier; pad the ends so a leading or trailing match still hits.
    padded = f".{name}."
    for old, new in _NAME_REWRITES:
        padded = padded.replace(old, new)
    name = padded[1:-1]
    if name.rsplit(".", 1)[-1] not in _SUPPORTED_TARGETS:
        return None
    return name if _MAPPED_TARGET_RE.match(name) else None


def _split_side(name: str) -> tuple[str, str] | None:
    """Split a tensor name into ``(raw_target, "a" | "b")``."""
    for marker, side in ((_LORA_A_MARKER, "a"), (_LORA_B_MARKER, "b")):
        index = name.find(marker)
        if index != -1:
            return name[:index], side
    return None


def _is_fasth3_release(weights_path: Path, config: Mapping[str, Any], has_gate: bool) -> bool:
    """Decide whether this adapter is a FastH3 four-step student.

    The step count is a hard contract - applying it to an adapter that was not
    distilled would silently truncate its schedule - so it is only applied to
    an adapter that identifies itself. A VSA compression gate is the release's
    own marker: FastVideo's runner reads the same payload to choose its
    attention backend. Otherwise the release line has to be named, which is how
    the published repositories and checkpoint files are spelled.
    """
    if has_gate:
        return True
    if config.get("_fasth3") is not None:
        return True
    haystack = "/".join(part.lower() for part in (*weights_path.parts[-3:], weights_path.stem))
    return "fasth3" in haystack.replace("-", "").replace("_", "")


def _read_base_schedule(config: Mapping[str, Any]) -> DMD2SigmaSchedule | None:
    marker = config.get("_fasth3")
    if isinstance(marker, Mapping):
        schedule = DMD2SigmaSchedule.from_metadata(marker)
        if schedule is not None:
            return schedule
    return DMD2SigmaSchedule.from_metadata(config)


def _infer_rank(tensors: Mapping[str, torch.Tensor]) -> int:
    """Rank is the shared inner dimension of every A/B pair."""
    ranks = {int(tensor.shape[0]) for name, tensor in tensors.items() if _LORA_A_MARKER in name}
    if len(ranks) != 1:
        raise ValueError(f"MiniMax-H3 LoRA mixes ranks across targets: {sorted(ranks)}")
    return ranks.pop()


def load_minimax_h3_fasth3_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper, FastH3Adapter | None] | None:
    """Load a PEFT-style LoRA over the MiniMax-H3 DiT.

    Returns ``None`` when the path is not such an adapter, which lets the
    caller fall through to the generic checkpoint loader. The third element is
    the FastH3 contract, present only for an adapter that identifies itself as
    a four-step student.
    """
    resolved = _resolve_adapter_files(lora_path)
    if resolved is None:
        return None
    weights_path, config_path = resolved
    config = _read_config(config_path)

    tensors: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    gate_tensors: list[str] = []
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        names = list(checkpoint.keys())
        for name in names:
            if _GATE_MARKER in name:
                gate_tensors.append(name)
                continue
            split = _split_side(name)
            if split is None:
                unmapped.append(name)
                continue
            raw_target, side = split
            module_name = _map_module_name(raw_target)
            if module_name is None:
                unmapped.append(name)
                continue
            tensor = checkpoint.get_tensor(name)
            if tensor.ndim != 2:
                raise ValueError(f"MiniMax-H3 LoRA tensors must be matrices, got {name}={tuple(tensor.shape)}")
            if side == "b" and module_name.endswith(".mlp.fc1"):
                # H3 stores the fused MLP projection gate-first; diffusers-style
                # exports carry it value-first.
                value, gate = tensor.chunk(2, dim=0)
                tensor = torch.cat((gate, value), dim=0).contiguous()
            tensors[f"{module_name}.lora_{side.upper()}.weight"] = tensor

    if not tensors:
        # Nothing here targets the H3 DiT, so this is somebody else's adapter.
        return None
    if unmapped:
        raise ValueError(
            f"MiniMax-H3 LoRA at {weights_path} has {len(unmapped)} tensors that do not name a "
            f"supported target: {sorted(unmapped)[:5]}"
        )

    incomplete = sorted(
        module_name
        for module_name in {name.rsplit(".lora_", 1)[0] for name in tensors}
        if f"{module_name}.lora_A.weight" not in tensors or f"{module_name}.lora_B.weight" not in tensors
    )
    if incomplete:
        raise ValueError(f"MiniMax-H3 LoRA has unpaired A/B tensors: {incomplete[:5]}")

    is_fasth3 = _is_fasth3_release(weights_path, config, bool(gate_tensors))
    if is_fasth3:
        if partition == "ref2va":
            raise ValueError("FastH3 preview v1 distills T2VA only, so it cannot run on a Ref2VA partition")
        if unsupported_offload_mode is not None:
            raise ValueError(f"FastH3 dynamic LoRA does not support {unsupported_offload_mode}")
    if gate_tensors:
        logger.warning(
            "FastH3 adapter at %s carries %d Video Sparse Attention gate tensors, which vLLM-Omni cannot "
            "yet apply to H3's packed [text | cond | audio | video] sequence. They are skipped and the "
            "adapter runs dense, matching FastVideo's --no-vsa mode. For dense serving prefer the Dense "
            "variant of the release.",
            weights_path,
            len(gate_tensors),
        )

    # Full-weight modules would be dropped by the LoRA path rather than
    # applied, so an adapter that ships them is refused instead of half-loaded.
    modules_to_save = config.get("modules_to_save")
    if modules_to_save:
        raise ValueError(f"MiniMax-H3 LoRA carries unsupported modules_to_save: {sorted(modules_to_save)[:5]}")

    rank = int(config.get("r") or _infer_rank(tensors))
    alpha = float(config.get("lora_alpha", rank))
    peft_helper = PEFTHelper.from_dict(
        {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": _TARGET_PATTERN,
            # rsLoRA scales by alpha/sqrt(r) rather than alpha/r, so it has to
            # be carried across or the adapter is applied at the wrong strength.
            "use_rslora": bool(config.get("use_rslora", False)),
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
    )
    pack_minimax_h3_fc1(lora_model)

    adapter = (
        FastH3Adapter(
            base_schedule=_read_base_schedule(config),
            vsa_gate_skipped=bool(gate_tensors),
        )
        if is_fasth3
        else None
    )
    logger.info(
        "Loaded MiniMax-H3 LoRA from %s: rank=%d, alpha=%g, targets=%d, fasth3=%s",
        weights_path,
        rank,
        alpha,
        len(lora_model.loras),
        is_fasth3,
    )
    return lora_model, peft_helper, adapter


__all__ = [
    "FASTH3_SIGMA_POINTS",
    "FASTH3_SUPPORTED_TASKS",
    "FastH3Adapter",
    "load_minimax_h3_fasth3_lora",
]
