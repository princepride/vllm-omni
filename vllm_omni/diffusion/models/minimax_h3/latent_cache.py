# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-addressed H3 clip latents for continuation across requests.

The video server tags each H3 request with ``extra_args["latent_cache_key"]``
(its job id). A finished request stores its denoised video/audio latents here,
on CPU. A later request naming that key in ``extra_args["continue_from"]``
samples a fresh window whose first ``continue_overlap_frames`` frames are
guided by the cached tail -- the ``latent_guide`` rows that in-request
continuation uses -- so no clip is decoded and re-encoded between requests.

Every DiT rank runs the same forward on identical inputs, so each rank stores
an identical copy and nothing is broadcast. The cache is per worker process:
it does not survive a restart and is not shared between data-parallel replicas.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.long_video import MINIMAX_H3_MAX_CONTINUATION_SECONDS

from .continuation import ContinuationWindow, _video_t

LATENT_CACHE_KEY = "latent_cache_key"
CONTINUE_FROM = "continue_from"
CONTINUE_OVERLAP_FRAMES = "continue_overlap_frames"
DEFAULT_CONTINUE_OVERLAP_FRAMES = 22
LATENT_CACHE_CLIPS_ENV = "VLLM_OMNI_H3_LATENT_CACHE_CLIPS"
DEFAULT_LATENT_CACHE_CLIPS = 16
_FPS = 24
_MAX_TIMELINE_FRAMES = int(MINIMAX_H3_MAX_CONTINUATION_SECONDS * _FPS)


@dataclass(frozen=True)
class CachedClip:
    """Denoised latents of one clip, placed on the chain's global frame timeline."""

    video: torch.Tensor  # [1, C, latent_t, latent_h, latent_w]
    audio: torch.Tensor  # [2, C, audio_t]
    start_frame: int
    num_frames: int

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.num_frames


class LatentClipCache:
    """Thread-safe LRU of :class:`CachedClip` keyed by video job id."""

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is None:
            capacity = int(os.environ.get(LATENT_CACHE_CLIPS_ENV, DEFAULT_LATENT_CACHE_CLIPS))
        if capacity < 0:
            raise ValueError(f"{LATENT_CACHE_CLIPS_ENV} must be non-negative")
        self.capacity = capacity
        self._clips: OrderedDict[str, CachedClip] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._clips)

    def put(self, key: str, clip: CachedClip) -> None:
        if self.capacity == 0:
            return
        with self._lock:
            self._clips[key] = clip
            self._clips.move_to_end(key)
            while len(self._clips) > self.capacity:
                self._clips.popitem(last=False)

    def get(self, key: str) -> CachedClip | None:
        with self._lock:
            clip = self._clips.get(key)
            if clip is not None:
                self._clips.move_to_end(key)
            return clip

    def clear(self) -> None:
        with self._lock:
            self._clips.clear()


@dataclass(frozen=True)
class LatentContinuationRequest:
    source: str
    overlap_frames: int


def resolve_latent_continuation(extra: Mapping[str, Any]) -> LatentContinuationRequest | None:
    source = extra.get(CONTINUE_FROM)
    if source is None:
        return None
    if not isinstance(source, str) or not source:
        raise OmniClientError("MiniMax H3 continue_from must be a non-empty video id")
    overlap = extra.get(CONTINUE_OVERLAP_FRAMES, DEFAULT_CONTINUE_OVERLAP_FRAMES)
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 5 or overlap % 17 != 5:
        raise OmniClientError("MiniMax H3 continue_overlap_frames must be an integer on the 17n+5 frame grid")
    return LatentContinuationRequest(source, overlap)


@dataclass(frozen=True)
class LatentContinuation:
    """A new window guided by the cached tail of the clip it continues."""

    window: ContinuationWindow
    video_tail: torch.Tensor
    audio_tail: torch.Tensor

    @property
    def overlap_v(self) -> int:
        return _video_t(self.window.overlap)


def plan_latent_continuation(
    clip: CachedClip,
    request: LatentContinuationRequest,
    *,
    num_frames: int,
    latent_h: int,
    latent_w: int,
) -> LatentContinuation:
    """Place ``num_frames`` new window frames so its first ``overlap`` frames repeat the clip tail."""
    overlap = request.overlap_frames
    if tuple(clip.video.shape[-2:]) != (latent_h, latent_w):
        raise OmniClientError(
            "MiniMax H3 continue_from requires the same canvas as the source video: "
            f"source latent {tuple(clip.video.shape[-2:])}, request latent {(latent_h, latent_w)}"
        )
    if overlap > clip.num_frames:
        raise OmniClientError(
            f"MiniMax H3 continue_overlap_frames ({overlap}) exceeds the source video ({clip.num_frames} frames)"
        )
    if overlap >= num_frames:
        raise OmniClientError(
            f"MiniMax H3 continuation window ({num_frames} frames) must exceed continue_overlap_frames ({overlap})"
        )
    start = clip.end_frame - overlap
    window = ContinuationWindow(start, start + num_frames, overlap)
    if window.end > _MAX_TIMELINE_FRAMES:
        raise OmniClientError(
            f"MiniMax H3 continuation chains are limited to {MINIMAX_H3_MAX_CONTINUATION_SECONDS:g} seconds"
        )
    overlap_audio_t = window.overlap_audio_t
    if overlap_audio_t > clip.audio.shape[-1]:
        raise OmniClientError("MiniMax H3 cached source audio is shorter than the requested overlap")
    return LatentContinuation(
        window=window,
        video_tail=clip.video[:, :, -_video_t(overlap) :],
        audio_tail=clip.audio[..., -overlap_audio_t:],
    )


def trim_decoded_overlap(
    video: torch.Tensor,
    audio: torch.Tensor,
    continuation: LatentContinuation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the hidden overlap from decoded ``[B, C, F, H, W]`` frames and ``[..., S]`` audio."""
    window = continuation.window
    window_audio_t = window.audio_end - window.audio_start
    drop_samples = round(audio.shape[-1] * window.overlap_audio_t / window_audio_t)
    return video[:, :, window.overlap :], audio[..., drop_samples:]
