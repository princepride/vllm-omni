# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.continuation import _audio_t, _video_t
from vllm_omni.diffusion.models.minimax_h3.latent_cache import (
    CachedClip,
    LatentClipCache,
    LatentContinuationRequest,
    plan_latent_continuation,
    resolve_latent_continuation,
    trim_decoded_overlap,
)
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_pack_audio_latent,
    minimax_h3_patchify_video_latent,
)
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

LATENT_H, LATENT_W = 4, 6


def _clip(num_frames: int, *, start: int = 0, fill: float = 0.0) -> CachedClip:
    latent_t = _video_t(num_frames)
    audio_t = _audio_t(start + num_frames) - _audio_t(start)
    video = fill + torch.arange(latent_t, dtype=torch.float32).view(1, 1, latent_t, 1, 1).expand(
        1, 24, latent_t, LATENT_H, LATENT_W
    )
    audio = fill + torch.arange(audio_t, dtype=torch.float32).view(1, 1, audio_t).expand(2, 32, audio_t)
    return CachedClip(video=video.contiguous(), audio=audio.contiguous(), start_frame=start, num_frames=num_frames)


def test_lru_refreshes_on_get_and_evicts_the_oldest():
    cache = LatentClipCache(capacity=2)
    first, second, third = (_clip(22, fill=i) for i in range(3))
    cache.put("a", first)
    cache.put("b", second)
    assert cache.get("a") is first
    cache.put("c", third)
    assert cache.get("b") is None
    assert cache.get("a") is first and cache.get("c") is third
    assert len(cache) == 2


def test_zero_capacity_disables_the_cache(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_H3_LATENT_CACHE_CLIPS", "0")
    cache = LatentClipCache()
    cache.put("a", _clip(22))
    assert cache.get("a") is None and len(cache) == 0


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({}, None),
        ({"continue_from": "video_gen_a"}, LatentContinuationRequest("video_gen_a", 22)),
        ({"continue_from": "v", "continue_overlap_frames": 39}, LatentContinuationRequest("v", 39)),
    ],
)
def test_resolve_latent_continuation(extra, expected):
    assert resolve_latent_continuation(extra) == expected


@pytest.mark.parametrize(
    "extra",
    [
        {"continue_from": ""},
        {"continue_from": 3},
        {"continue_from": "v", "continue_overlap_frames": 20},
        {"continue_from": "v", "continue_overlap_frames": True},
        {"continue_from": "v", "continue_overlap_frames": "22"},
    ],
)
def test_invalid_latent_continuation_options(extra):
    with pytest.raises(OmniClientError):
        resolve_latent_continuation(extra)


@pytest.mark.parametrize("source_start", [0, 102])
def test_window_continues_the_global_timeline_from_the_cached_tail(source_start):
    clip = _clip(124, start=source_start)
    plan = plan_latent_continuation(
        clip, LatentContinuationRequest("a", 22), num_frames=141, latent_h=LATENT_H, latent_w=LATENT_W
    )
    window = plan.window
    assert (window.start, window.end, window.overlap) == (clip.end_frame - 22, clip.end_frame + 119, 22)
    assert plan.overlap_v == 7
    torch.testing.assert_close(plan.video_tail, clip.video[:, :, -7:])
    # Audio boundaries are taken on the cumulative frame timeline.
    overlap_audio_t = _audio_t(clip.end_frame) - _audio_t(window.start)
    assert window.overlap_audio_t == overlap_audio_t
    torch.testing.assert_close(plan.audio_tail, clip.audio[..., -overlap_audio_t:])


@pytest.mark.parametrize(
    "clip,num_frames,overlap,latent_hw,match",
    [
        (_clip(124), 141, 22, (LATENT_H, LATENT_W + 2), "same canvas"),
        (_clip(22), 141, 39, (LATENT_H, LATENT_W), "exceeds the source video"),
        (_clip(124), 22, 22, (LATENT_H, LATENT_W), "must exceed"),
        (_clip(124, start=7100), 141, 22, (LATENT_H, LATENT_W), "limited to 300 seconds"),
    ],
)
def test_invalid_continuation_windows(clip, num_frames, overlap, latent_hw, match):
    with pytest.raises(OmniClientError, match=match):
        plan_latent_continuation(
            clip,
            LatentContinuationRequest("a", overlap),
            num_frames=num_frames,
            latent_h=latent_hw[0],
            latent_w=latent_hw[1],
        )


def test_trim_drops_the_hidden_overlap_frames_and_audio():
    plan = plan_latent_continuation(
        _clip(124), LatentContinuationRequest("a", 22), num_frames=141, latent_h=LATENT_H, latent_w=LATENT_W
    )
    window_audio_t = plan.window.audio_end - plan.window.audio_start
    video = torch.arange(141, dtype=torch.float32).view(1, 1, 141, 1, 1).expand(1, 3, 141, 2, 2)
    audio = torch.arange(window_audio_t * 800, dtype=torch.float32).view(1, 1, -1).expand(1, 2, -1)
    trimmed_video, trimmed_audio = trim_decoded_overlap(video, audio, plan)
    assert trimmed_video.shape[2] == 119 and trimmed_video[0, 0, 0, 0, 0] == 22
    assert trimmed_audio.shape[-1] == (window_audio_t - plan.window.overlap_audio_t) * 800
    assert trimmed_audio[0, 0, 0] == plan.window.overlap_audio_t * 800


# ---------------------------------------------------------------- pipeline


def _conditioning(task: str, num_frames: int):
    from vllm_omni.model_executor.models.minimax_h3.conditioning import (
        MiniMaxH3EncoderConditioning,
        MiniMaxH3EncoderMediaConditioning,
        MiniMaxH3TextConditioning,
    )

    media = MiniMaxH3EncoderMediaConditioning(
        task=task,
        height=LATENT_H * 16,
        width=LATENT_W * 16,
        num_frames=num_frames,
        latent_t=_video_t(num_frames),
        audio_t=_audio_t(num_frames),
    )
    text = MiniMaxH3TextConditioning(torch.zeros(3, 5120, dtype=torch.bfloat16), torch.ones(3, dtype=torch.int64))
    return MiniMaxH3EncoderConditioning.from_components(text, media)


def _pipeline(*, cache_dit=None):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.od_config = SimpleNamespace(step_execution=False)
    pipeline.load_vae_decoder = True
    pipeline._fasth3 = None
    pipeline._fasth3_checkpoint = None
    pipeline._vdn = None
    pipeline._active_turbo_spec = lambda _sampling: None
    pipeline._has_active_native_lora = lambda _sampling: False
    pipeline._resolve_task = lambda task, **_kwargs: task
    pipeline._resolve_sigma_positions = lambda _task, _sampling: (None, 3)
    pipeline._quality_policy = SimpleNamespace(resolve=lambda **_kwargs: SimpleNamespace(cache_dit=cache_dit))
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda _plan: None)
    pipeline._latent_clip_cache = LatentClipCache(capacity=4)
    pipeline._offload_model_cpu_stage_output = lambda value: value
    pipeline._release_stage_cache = lambda: None
    calls = []

    def diffuse(**kwargs):
        calls.append(kwargs)
        latent_t, audio_t = kwargs["latent_t"], kwargs["audio_t"]
        # Distinct values per call and per latent frame expose a wrong tail slice.
        video = 100.0 * len(calls) + torch.arange(latent_t, dtype=torch.float32).view(1, 1, -1, 1, 1)
        audio = 100.0 * len(calls) + torch.arange(audio_t, dtype=torch.float32).view(1, 1, -1)
        return (
            video.expand(1, 24, latent_t, kwargs["latent_h"], kwargs["latent_w"]).contiguous(),
            audio.expand(2, 32, audio_t).contiguous(),
        )

    def decode(video_latent, audio_latent, *, height, width):
        frames = (video_latent.shape[2] - 2) // 5 * 17 + 5
        video = torch.arange(frames, dtype=torch.float32).view(1, 1, -1, 1, 1) / frames
        audio = torch.arange(audio_latent.shape[-1] * 800, dtype=torch.float32).view(1, 1, -1)
        return video.expand(1, 3, frames, height, width).contiguous(), audio.expand(1, 2, -1).contiguous()

    pipeline.diffuse = diffuse
    pipeline.decode = decode
    return pipeline, calls


def _run(pipeline, conditioning, extra, *, num_outputs=1):
    sampling = SimpleNamespace(extra_args=extra, quality=None, seed=7, num_outputs_per_prompt=num_outputs)
    pipeline._prepare_request_inputs = lambda _prompt, _sampling: pipeline._prepare_encoder_conditioning_inputs(
        conditioning, _sampling
    )
    output = pipeline.forward(SimpleNamespace(prompts=["prompt"], sampling_params=sampling))
    return output.output


def test_requests_chain_through_cached_latent_tails():
    pipeline, calls = _pipeline()
    first_video, first_audio = _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "video_a"})
    assert first_video.shape[1] == 124
    cached = pipeline._latent_clip_cache.get("video_a")
    assert (cached.start_frame, cached.num_frames) == (0, 124)
    assert "ref_blocks" not in calls[0] or not calls[0]["ref_blocks"]

    video, audio = _run(
        pipeline, _conditioning("ref2va", 141), {"latent_cache_key": "video_b", "continue_from": "video_a"}
    )
    guided = calls[1]
    start, end = 102, 243
    assert guided["temporal_offset"] == pytest.approx(start * 40 / 24)
    assert guided["audio_t"] == _audio_t(end) - _audio_t(start)
    overlap_audio_t = _audio_t(124) - _audio_t(start)
    assert guided["ref_blocks"] == [
        {
            "kind": "latent_guide",
            "latent_t": 7,
            "latent_h": LATENT_H,
            "latent_w": LATENT_W,
            "ref_audio_t": overlap_audio_t,
        }
    ]
    torch.testing.assert_close(
        guided["visual_condition"],
        minimax_h3_patchify_video_latent(cached.video[:, :, -7:], patch_size=(1, 2, 2)),
    )
    torch.testing.assert_close(
        guided["audio_condition"], minimax_h3_pack_audio_latent(cached.audio[..., -overlap_audio_t:])
    )
    # Only the new frames are returned; the guided overlap stays hidden.
    assert video.shape[1] == 141 - 22
    assert audio.shape[-1] == (guided["audio_t"] - overlap_audio_t) * 800

    continued = pipeline._latent_clip_cache.get("video_b")
    assert (continued.start_frame, continued.num_frames) == (start, 141)
    _run(pipeline, _conditioning("ref2va", 107), {"latent_cache_key": "video_c", "continue_from": "video_b"})
    assert calls[2]["temporal_offset"] == pytest.approx((end - 22) * 40 / 24)
    torch.testing.assert_close(
        calls[2]["visual_condition"],
        minimax_h3_patchify_video_latent(continued.video[:, :, -7:], patch_size=(1, 2, 2)),
    )


def test_unknown_source_is_a_client_error():
    pipeline, _ = _pipeline()
    with pytest.raises(OmniClientError, match="no cached latents"):
        _run(pipeline, _conditioning("ref2va", 141), {"continue_from": "video_missing"})


@pytest.mark.parametrize(
    "task,extra,num_outputs,cache_dit,match",
    [
        ("t2va", {}, 1, None, "requires the Ref2VA task"),
        ("ref2va", {"preencode_mp4": True}, 1, None, "preencode_mp4"),
        ("ref2va", {}, 2, None, "multiple outputs"),
        ("ref2va", {}, 1, object(), "quality=lossless"),
        ("ref2va", {"long_video": True, "long_video_mode": "continuation"}, 1, None, "long_video_mode=continuation"),
    ],
)
def test_rejected_continuation_combinations(task, extra, num_outputs, cache_dit, match):
    pipeline, _ = _pipeline(cache_dit=cache_dit)
    _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "video_a"})
    with pytest.raises(OmniClientError, match=match):
        _run(pipeline, _conditioning(task, 141), {"continue_from": "video_a", **extra}, num_outputs=num_outputs)


def test_only_tagged_single_output_requests_are_cached():
    pipeline, _ = _pipeline()
    _run(pipeline, _conditioning("t2va", 124), {})
    _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "video_two"}, num_outputs=2)
    assert len(pipeline._latent_clip_cache) == 0


# ---------------------------------------------------------------- encoder


def test_continuation_defaults_to_ref2va_without_visual_references():
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
        _resolve_task,
        _validate_ref2va_reference_counts,
    )

    assert _resolve_task({"continue_from": "v"}, {}) == "ref2va"
    assert _resolve_task({"continue_from": "v", "task": "T2VA"}, {}) == "t2va"
    _validate_ref2va_reference_counts(0, 0, 1, latent_continuation=True)
    with pytest.raises(OmniClientError, match="at least one image or video"):
        _validate_ref2va_reference_counts(0, 0, 1)


def test_encoder_rejects_continue_from_on_other_tasks():
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import prepare_encoder_inputs

    with pytest.raises(OmniClientError, match="requires the Ref2VA task"):
        prepare_encoder_inputs(
            {"prompt": "the scene continues"},
            SimpleNamespace(extra_args={"continue_from": "v", "task": "t2va", "aspect_ratio": "16:9"}),
        )
