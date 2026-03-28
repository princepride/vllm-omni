# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Ported from daVinci-MagiHuman inference/pipeline/video_process.py (audio parts)

from __future__ import annotations

import math

import torch
import whisper

_SAMPLE_RATE = 51200
_AUDIO_CHUNK_DURATION = 29
_OVERLAP_RATIO = 0.5


def merge_overlapping_vae_features(audio_feats: list[torch.Tensor], overlap_ratio: float = 0.5) -> torch.Tensor | None:
    if not audio_feats:
        return None
    if len(audio_feats) == 1:
        return audio_feats[0]

    batch_size, total_frames, feature_dim = audio_feats[0].shape
    overlap_frames = int(total_frames * overlap_ratio)
    step_frames = total_frames - overlap_frames
    final_length = (len(audio_feats) - 1) * step_frames + total_frames
    output_feat = torch.zeros(
        batch_size, final_length, feature_dim, device=audio_feats[0].device, dtype=audio_feats[0].dtype
    )

    for block_idx, current_feat in enumerate(audio_feats):
        output_start = block_idx * step_frames
        if block_idx == 0:
            output_feat[:, output_start : output_start + total_frames, :] = current_feat
            continue

        non_overlap_start = output_start + overlap_frames
        non_overlap_end = output_start + total_frames
        output_feat[:, non_overlap_start:non_overlap_end, :] = current_feat[:, overlap_frames:, :]

        for frame_idx in range(overlap_frames):
            output_pos = output_start + frame_idx
            prev_weight = (overlap_frames - frame_idx) / overlap_frames
            curr_weight = frame_idx / overlap_frames
            output_feat[:, output_pos, :] = (
                prev_weight * output_feat[:, output_pos, :] + curr_weight * current_feat[:, frame_idx, :]
            )
    return output_feat


def load_audio_and_encode(audio_vae, audio_path: str, seconds: int | None = None) -> torch.Tensor:
    """Load audio from file and encode to latent space using the Stable Audio VAE."""
    audio_full = whisper.load_audio(audio_path, sr=_SAMPLE_RATE)
    if seconds is not None:
        audio_full = audio_full[: min(int(seconds * _SAMPLE_RATE), audio_full.shape[0])]
    total_samples = audio_full.shape[0]

    window_size = int(_AUDIO_CHUNK_DURATION * _SAMPLE_RATE)
    step_size = int(window_size * (1 - _OVERLAP_RATIO))
    if total_samples <= window_size:
        audio = torch.from_numpy(audio_full).cuda()
        audio = audio.unsqueeze(0).expand(2, -1)
        return audio_vae.vae_model.encode(audio)

    encoded_chunks = []
    latent_to_audio_ratio = None
    for offset_start in range(0, total_samples, step_size):
        offset_end = min(offset_start + window_size, total_samples)
        chunk = whisper.pad_or_trim(audio_full[offset_start:offset_end], length=window_size)
        chunk_tensor = torch.from_numpy(chunk).cuda().unsqueeze(0).expand(2, -1)
        encoded_chunk = audio_vae.vae_model.encode(chunk_tensor)

        if latent_to_audio_ratio is None:
            latent_to_audio_ratio = encoded_chunk.shape[-1] / window_size

        encoded_chunks.append(encoded_chunk.permute(0, 2, 1))
        if offset_end >= total_samples:
            break

    final_feat = merge_overlapping_vae_features(encoded_chunks, overlap_ratio=_OVERLAP_RATIO).permute(0, 2, 1)
    final_target_len = math.ceil(total_samples * latent_to_audio_ratio)
    return final_feat[:, :, :final_target_len]
