# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FastVideo FastH3: a four-step DMD2 student of MiniMax-H3.

FastH3 replaces H3's 49 denoiser evaluations with four. It ships as an adapter
over the base checkpoint rather than as a full release, so it reuses H3's text
encoder, video VAE, audio VAE, tokenizers and schedulers unchanged.

The artifact is *not* a PEFT LoRA, and it is not request-switchable. Its own
metadata states the reconstruction as::

    W = W_base + lora_B @ lora_A; then .diff/.diff_b added and .set_weight assigned

so besides rank-64 factors it carries full-rank ``.diff``/``.diff_b`` deltas for
RMSNorm weights, biases, patch projections and the final layer - none of which a
LoRA layer can express - and the VSA variants add ``.set_weight`` tensors for
compression gates that do not exist in the base transformer at all. The adapter
is therefore fused into the checkpoint stream at load time, before the weights
are sharded, which is also what the release's model card requires.

The low-rank factors carry no alpha: the reconstruction adds ``lora_B @ lora_A``
directly, i.e. a scale of exactly 1.

Two checkpoint spellings meet here. The adapter is written in the diffusers
namespace (``transformer_blocks.0.attn.to_q``) while vLLM-Omni loads H3's native
one (``blocks.0.attn.qkv_proj``), whose attention and MLP projections are fused.
Every mapping and layout convention below was verified tensor by tensor against
the released full checkpoint (``FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree``):
``W_base + delta`` reproduces it to bf16 rounding.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

FASTH3_FORMAT = "fastvideo-lora-v2"
FASTH3_MANIFEST = "adapter_manifest.json"

# Five scheduler points bound the four transformer forwards the student was
# trained for. The ladder is the release's `dmd_denoising_steps` divided by
# 1000; it starts at 0.999 rather than 1.0 because training capped the noise
# level there (`max_timestep_ratio`), and it is uniform, so no rectified-flow
# time shift is applied on top of it.
FASTH3_SIGMA_POINTS = 5
FASTH3_FLOW_SHIFT = 1.0
# Preview v1 distills the text-to-video-and-audio path only.
FASTH3_SUPPORTED_TASKS = frozenset({"t2va"})

_LORA_A = ".lora_A.weight"
_LORA_B = ".lora_B.weight"
_DIFF = ".diff"
_DIFF_B = ".diff_b"
_SET_WEIGHT = ".set_weight"

# Adapter module prefix -> the native parameter it edits, minus the
# ``.weight``/``.bias`` suffix.
_MODEL_LEVEL_TARGETS = {
    "proj_in": "video_patch_proj",
    "proj_out": "final_layer.video_out",
    "audio_proj_in": "audio_patch_proj",
    "audio_proj_out": "final_layer.audio_out",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "norm_out.norm": "final_layer.norm",
}

# Per-block adapter suffix -> (native suffix, how a delta enters the native
# parameter). H3 stores attention as one grouped QKV matrix and the MLP as one
# fused gate/up matrix, so those deltas need placing rather than adding.
_PLAIN, _QKV_Q, _QKV_K, _QKV_V, _SWAP_HALVES = "plain", "q", "k", "v", "swap_halves"
_QKV_SLOTS = (_QKV_Q, _QKV_K, _QKV_V)
_BLOCK_TARGETS = {
    "attn.to_q": ("attn.qkv_proj", _QKV_Q),
    "attn.to_k": ("attn.qkv_proj", _QKV_K),
    "attn.to_v": ("attn.qkv_proj", _QKV_V),
    "attn.to_out.0": ("attn.out_proj", _PLAIN),
    "ff.net.0.proj": ("mlp.fc1", _SWAP_HALVES),
    "ff.net.2": ("mlp.fc2", _PLAIN),
    "adaln_proj.linear": ("adaln_proj.linear", _PLAIN),
    "norm1": ("norm1", _PLAIN),
    "norm2": ("norm2", _PLAIN),
}

# Adapter block prefix -> native block prefix.
_BLOCK_PREFIXES = (
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("transformer_blocks.", "blocks."),
)


class FastH3AdapterError(ValueError):
    """The artifact is a FastH3 adapter, but it cannot be applied as one."""


@dataclass
class _ParamPatch:
    """Everything the adapter contributes to one native parameter."""

    # layout -> (lora_A, lora_B). A grouped QKV parameter collects three.
    low_rank: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = field(default_factory=dict)
    diff: torch.Tensor | None = None
    layout: str = _PLAIN


def _swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    """Exchange the two halves of a fused gate/up matrix.

    The diffusers export stores the feed-forward projection value-first while
    H3's native ``mlp.fc1`` is gate-first, so a delta computed in the diffusers
    layout has to be swapped before it can be added to the native parameter.
    """
    if tensor.shape[0] % 2:
        raise FastH3AdapterError(f"fused gate/up delta must split evenly, got {tuple(tensor.shape)}")
    first, second = tensor.chunk(2, dim=0)
    return torch.cat((second, first), dim=0)


def _place_in_grouped_qkv(deltas: Mapping[str, torch.Tensor], *, head_dim: int) -> torch.Tensor:
    """Interleave per-projection deltas into H3's grouped QKV layout.

    The checkpoint stores one head group at a time as ``[q, k, v]``, which is
    what :func:`_reorder_grouped_qkv_to_qkv` unpacks on the way in. A delta
    built from the separate diffusers projections has to be folded back into
    that order.
    """
    missing = sorted(set(_QKV_SLOTS) - set(deltas))
    if missing:
        raise FastH3AdapterError(f"grouped QKV delta is missing its {missing} projections")
    parts = []
    for slot in _QKV_SLOTS:
        delta = deltas[slot]
        if delta.shape[0] % head_dim:
            raise FastH3AdapterError(
                f"QKV {slot} delta rows {delta.shape[0]} are not a multiple of head_dim {head_dim}"
            )
        parts.append(delta.reshape(delta.shape[0] // head_dim, head_dim, *delta.shape[1:]))
    groups = parts[0].shape[0]
    if any(part.shape[0] != groups for part in parts):
        raise FastH3AdapterError("QKV projections disagree on the number of head groups")
    return torch.cat(parts, dim=1).reshape(groups * 3 * head_dim, *parts[0].shape[2:])


def _resolve_native_target(module: str) -> tuple[str, str] | None:
    """Map an adapter module path to ``(native module path, layout)``."""
    native = _MODEL_LEVEL_TARGETS.get(module)
    if native is not None:
        return native, _PLAIN
    for adapter_prefix, native_prefix in _BLOCK_PREFIXES:
        if not module.startswith(adapter_prefix):
            continue
        remainder = module[len(adapter_prefix) :]
        index, _, suffix = remainder.partition(".")
        if not index.isdigit():
            return None
        target = _BLOCK_TARGETS.get(suffix)
        if target is None:
            return None
        native_suffix, layout = target
        return f"{native_prefix}{index}.{native_suffix}", layout
    return None


def _split_adapter_key(name: str) -> tuple[str, str] | None:
    """Split an adapter tensor name into ``(module path, role)``."""
    for marker, role in (
        (_LORA_A, "lora_a"),
        (_LORA_B, "lora_b"),
        (_DIFF_B, "diff_b"),
        (_DIFF, "diff"),
        (_SET_WEIGHT, "set_weight"),
    ):
        if name.endswith(marker):
            return name[: -len(marker)], role
    return None


class FastH3WeightFusion:
    """Fuse a FastH3 adapter into the H3 checkpoint stream as it is loaded."""

    def __init__(
        self,
        *,
        source: Path,
        patches: Mapping[str, _ParamPatch],
        head_dim: int,
        requires_vsa: bool,
    ) -> None:
        self._source = source
        self._patches = dict(patches)
        self._head_dim = head_dim
        self.requires_vsa = requires_vsa
        self._applied: set[str] = set()
        self._device: torch.device | None = None

    @property
    def source(self) -> Path:
        return self._source

    @classmethod
    def from_path(cls, path: str | Path, *, head_dim: int) -> FastH3WeightFusion | None:
        """Build a fusion from an adapter file or directory, else ``None``.

        Returning ``None`` keeps every other ``--lora-path`` artifact on the
        dynamic LoRA route; only a ``fastvideo-lora-v2`` file is claimed here.
        """
        weights_path = _resolve_adapter_file(path)
        if weights_path is None:
            return None

        patches: dict[str, _ParamPatch] = {}
        gate_tensors: list[str] = []
        unmapped: list[str] = []
        with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if metadata.get("format") != FASTH3_FORMAT:
                return None
            for name in checkpoint.keys():
                split = _split_adapter_key(name)
                if split is None:
                    unmapped.append(name)
                    continue
                module, role = split
                if role == "set_weight":
                    # A VSA compression gate: a module the base transformer does
                    # not have, so there is nothing to fuse it into.
                    gate_tensors.append(name)
                    continue
                target = _resolve_native_target(module)
                if target is None:
                    unmapped.append(name)
                    continue
                native_module, layout = target
                native_param = f"{native_module}.{'bias' if role == 'diff_b' else 'weight'}"
                patch = patches.setdefault(native_param, _ParamPatch(layout=layout))
                tensor = checkpoint.get_tensor(name)
                if role in ("diff", "diff_b"):
                    if patch.diff is not None:
                        raise FastH3AdapterError(f"duplicate {role} for {native_param}")
                    patch.diff = tensor
                else:
                    a, b = patch.low_rank.get(layout, (None, None))
                    patch.low_rank[layout] = (tensor, b) if role == "lora_a" else (a, tensor)

        if unmapped:
            raise FastH3AdapterError(
                f"FastH3 adapter at {weights_path} has {len(unmapped)} tensors that name no known "
                f"H3 parameter: {sorted(unmapped)[:5]}"
            )
        for native_param, patch in patches.items():
            for slot, (a, b) in patch.low_rank.items():
                if a is None or b is None:
                    raise FastH3AdapterError(f"FastH3 adapter has an unpaired factor for {native_param} slot {slot!r}")
            # Only the low-rank factors are placed into H3's fused QKV and
            # gate/up layouts; a full-rank delta is added as it comes, so one
            # aimed at a fused parameter would silently land transposed.
            if patch.diff is not None and patch.layout != _PLAIN:
                raise FastH3AdapterError(
                    f"FastH3 adapter carries a full-rank delta for {native_param}, which H3 stores in the "
                    f"{patch.layout!r} fused layout; this loader can only place low-rank factors there"
                )

        fusion = cls(
            source=weights_path,
            patches=patches,
            head_dim=head_dim,
            requires_vsa=bool(gate_tensors),
        )
        logger.info(
            "FastH3 adapter %s: rank=%s, parameters patched=%d, low-rank=%s, diff=%s, set_weight=%d",
            weights_path,
            metadata.get("rank", "?"),
            len(patches),
            metadata.get("low_rank_tensors", "?"),
            metadata.get("diff_tensors", "?"),
            len(gate_tensors),
        )
        return fusion

    def _compute_device(self, weight: torch.Tensor) -> torch.device:
        """Where to reconstruct a delta.

        H3's per-block modulation projection is 96768x2688, so rebuilding all
        343 patched parameters is a few TFLOP of rank-64 products. On CPU that
        adds minutes to a load that already has a startup deadline, so the
        accelerator does the arithmetic whenever there is one.
        """
        if weight.device.type != "cpu":
            return weight.device
        if self._device is None:
            self._device = torch.device(torch.accelerator.current_accelerator() or "cpu")
        return self._device

    @staticmethod
    def _widen(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move to ``device``, then widen to float32.

        Asking ``Tensor.to`` for a device and a dtype at once converts on the
        host and ships twice the bytes; splitting it moves bfloat16 and widens
        on the accelerator.
        """
        return tensor.to(device, non_blocking=True).to(torch.float32)

    def fuse(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """Return ``weight`` with this adapter's contribution added."""
        patch = self._patches.get(name)
        if patch is None:
            return weight
        self._applied.add(name)

        device = self._compute_device(weight)

        delta: torch.Tensor | None = None
        if patch.low_rank:
            if patch.layout in _QKV_SLOTS:
                per_slot = {
                    slot: self._widen(b, device) @ self._widen(a, device) for slot, (a, b) in patch.low_rank.items()
                }
                delta = _place_in_grouped_qkv(per_slot, head_dim=self._head_dim)
            else:
                a, b = patch.low_rank[patch.layout]
                if patch.layout == _SWAP_HALVES:
                    # Permuting the rows of B permutes the rows of the product,
                    # so swap the rank-64 factor instead of the full delta.
                    b = _swap_halves(b)
                delta = self._widen(b, device) @ self._widen(a, device)
        if patch.diff is not None:
            diff = self._widen(patch.diff, device)
            delta = diff if delta is None else delta + diff
        if delta is None:
            return weight
        if delta.shape != weight.shape:
            raise FastH3AdapterError(
                f"FastH3 delta for {name} has shape {tuple(delta.shape)}, parameter is {tuple(weight.shape)}"
            )
        # Leave the result on the compute device. These weights are bound for
        # the accelerator anyway, so returning them to host memory would pay a
        # device-to-host copy of the whole checkpoint only for the loader to
        # send it straight back: measured at 152s against 15s for 60 GiB of
        # patched projections, against 17s for the unavoidable upload alone.
        # Fold the base weight into the freshly built delta in place. Promoting
        # the weight to float32 on its own would allocate two more buffers the
        # size of the parameter, and H3's largest patched projection is 0.5 GiB.
        return delta.add_(weight.to(device, non_blocking=True)).to(weight.dtype)

    def apply(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Fuse every streamed checkpoint tensor on its way into the model."""
        for name, weight in weights:
            yield name, self.fuse(name, weight)

    def validate_fully_applied(self) -> None:
        """Close the fusion: every edit must have met its parameter.

        A silently unapplied delta is the failure mode that matters here: the
        model would load and generate, just not as the distilled student. The
        weights are loaded once, so the mapped payloads are dropped afterwards
        rather than held for the life of the process.
        """
        missing = sorted(set(self._patches) - self._applied)
        if missing:
            raise FastH3AdapterError(
                f"FastH3 adapter edits {len(missing)} parameters the checkpoint never provided: {missing[:5]}"
            )
        for patch in self._patches.values():
            patch.low_rank.clear()
            patch.diff = None


def _resolve_adapter_file(path: str | Path) -> Path | None:
    """Find the single adapter file at ``path``, or ``None``."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate if candidate.suffix == ".safetensors" else None
    if not candidate.is_dir():
        return None
    named = candidate / "adapter_model.safetensors"
    if named.is_file():
        return named
    files = sorted(candidate.glob("*.safetensors"))
    if len(files) == 1:
        return files[0]
    # The published repository bundles four variants under one root, so a
    # directory holding several adapters is ambiguous rather than loadable.
    if len(files) > 1 or (candidate / FASTH3_MANIFEST).is_file():
        raise FastH3AdapterError(
            f"{candidate} holds {len(files)} adapters; point --lora-path at one variant "
            "(for example dense-datafree/adapter_model.safetensors)"
        )
    return None


__all__ = [
    "FASTH3_FLOW_SHIFT",
    "FASTH3_FORMAT",
    "FASTH3_SIGMA_POINTS",
    "FASTH3_SUPPORTED_TASKS",
    "FastH3AdapterError",
    "FastH3WeightFusion",
]
