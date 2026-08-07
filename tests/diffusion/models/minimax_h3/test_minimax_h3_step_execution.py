# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax H3 step-wise execution (continuous batching) contract tests.

These run on CPU against a stand-in DiT, so they cover the packing, the
per-request scheduler math, and the runner-facing state wiring without needing
checkpoint weights.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_HIDDEN = 8


class _SegmentMeanModel:
    """Stand-in DiT whose output depends on packed-document boundaries.

    Every row is mixed with the mean of its own ``cu_seqlens`` document and with
    its own timestep, so a wrong document boundary, row offset, or timestep
    assignment shows up as a numeric difference instead of passing silently.
    """

    def __call__(self, **kwargs):
        x = kwargs["x"][0]
        audio_x = kwargs["audio_x"][0]
        bounds = kwargs["packed_seq_params"]["cu_seqlens_q"].tolist()
        img_pos = kwargs["img_pos_info"]["position_ids"]
        audio_pos = kwargs["audio_pos_info"]["position_ids"]

        pooled_video = torch.zeros_like(x)
        pooled_audio = torch.zeros_like(audio_x)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop <= start:
                continue
            pooled_video[start:stop] = x[start:stop].mean(dim=0, keepdim=True)
            pooled_audio[start:stop] = audio_x[start:stop].mean(dim=0, keepdim=True)

        row_timesteps = kwargs["unique_timesteps"][kwargs["inverse_indices"]].unsqueeze(-1)
        video = (pooled_video + 0.5 * x + row_timesteps)[img_pos]
        audio = (pooled_audio + 0.5 * audio_x + row_timesteps)[audio_pos]
        if not kwargs.get("skip_mask_out_condition", False):
            video = video * kwargs["update_mask"].view(-1).unsqueeze(-1)
        return video, audio


def _make_branch(*, text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int, seed: int):
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence

    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    generator = torch.Generator().manual_seed(seed)
    text_embeddings = torch.randn(text_len, _HIDDEN, generator=generator, dtype=torch.float32)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=text_embeddings,
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )
    video_rows = torch.randn(int(branch.img_pos.shape[0]), 96, generator=generator, dtype=torch.float32)
    audio_rows = torch.randn(int(branch.audio_pos.shape[0]), 32, generator=generator, dtype=torch.float32)
    return branch, video_rows, audio_rows


def _sigmas(num_points: int, shift: float) -> list[float]:
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    return minimax_h3_time_shift_sigmas(num_steps=num_points, shift_scale=shift)


def _make_state(request_id: str, model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.worker.utils import StepRequestState

    state = StepRequestState(request_id=request_id, sampling=SimpleNamespace())
    state.latents = video_rows.clone()
    state.timesteps = torch.tensor([1.0 - sigma for sigma in sigmas_video[:-1]], dtype=torch.float32)
    state.step_index = 0
    state.extra = {
        mod._STEP_BRANCH: branch,
        # Co-batched requests must share one DiT instance, or denoise_step()
        # treats the batch as mixed-task and falls back to one forward each.
        mod._STEP_TRANSFORMER: model,
        mod._STEP_AUDIO_ROWS: audio_rows.clone(),
        mod._STEP_COND_ANCHOR: None,
        mod._STEP_AUDIO_ANCHOR: None,
        mod._STEP_SIGMAS_VIDEO: sigmas_video,
        mod._STEP_SIGMAS_AUDIO: sigmas_audio,
    }
    return state


def _step_pipeline(model, *, packed_batch_supported: bool = True):
    """A pipeline instance carrying only what the step methods touch."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.transformer = model
    pipeline.device = torch.device("cpu")
    pipeline._transformer_for_task = lambda task: model
    pipeline._packed_batch_supported = lambda transformer: packed_batch_supported
    return pipeline


def test_pipeline_declares_step_execution_contract():
    from vllm_omni.diffusion.models.interface import SupportsStepExecution
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    assert MiniMaxH3Pipeline.supports_step_execution is True
    for method in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        assert callable(getattr(MiniMaxH3Pipeline, method)), method
    # The runner gates step mode on this structural check.
    assert isinstance(_step_pipeline(_SegmentMeanModel()), SupportsStepExecution)


def test_single_request_batch_layout_is_the_request_mode_layout():
    """A batch of one must be the request-mode kwargs, not a lookalike.

    Rebuilding it would silently drop ``video_token_layout``, the hint sparse
    attention backends need to find the video segment.
    """
    from vllm_omni.diffusion.models.minimax_h3.step_batch import minimax_h3_batched_forward_kwargs

    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=0)
    request_kwargs = branch.forward_kwargs(
        video_rows=video_rows,
        audio_rows=audio_rows,
        t_video=0.25,
        t_audio=0.4,
        imgvid_cond_timestep=0.999,
        audio_ref_cond_timestep=1.0,
    )
    batched_kwargs = minimax_h3_batched_forward_kwargs(
        branches=[branch],
        video_rows=[video_rows],
        audio_rows=[audio_rows],
        t_video=[0.25],
        t_audio=[0.4],
        imgvid_cond_timesteps=[0.999],
        audio_ref_cond_timesteps=[1.0],
    )

    assert batched_kwargs.keys() == request_kwargs.keys()
    assert batched_kwargs["video_token_layout"] == request_kwargs["video_token_layout"]
    for key, expected in request_kwargs.items():
        actual = batched_kwargs[key]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected)
        elif isinstance(expected, dict):
            for field, value in expected.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(actual[field], value)
                else:
                    assert actual[field] == value, key
        else:
            assert actual == expected, key


def test_batched_layout_keeps_one_document_per_request():
    from vllm_omni.diffusion.models.minimax_h3.step_batch import minimax_h3_batched_forward_kwargs

    first, first_video, first_audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=1)
    second, second_video, second_audio = _make_branch(text_len=5, latent_t=3, latent_h=6, latent_w=4, audio_t=2, seed=2)

    kwargs = minimax_h3_batched_forward_kwargs(
        branches=[first, second],
        video_rows=[first_video, second_video],
        audio_rows=[first_audio, second_audio],
        t_video=[0.25, 0.5],
        t_audio=[0.4, 0.6],
        imgvid_cond_timesteps=[0.999, 0.999],
        audio_ref_cond_timesteps=[1.0, 1.0],
    )

    bounds = kwargs["packed_seq_params"]["cu_seqlens_q"].tolist()
    assert bounds == [
        0,
        first.used_len,
        first.seq_len,
        first.seq_len + second.used_len,
        first.seq_len + second.seq_len,
    ]
    assert kwargs["x"].shape == (1, first.seq_len + second.seq_len, 96)
    assert kwargs["refiner_packed_seq_params"]["cu_seqlens_q"].tolist() == [
        0,
        first.text_len,
        first.text_len + second.text_len,
    ]

    # The second request's rows must live entirely past the first request's block.
    second_img_pos = kwargs["img_pos_info"]["position_ids"][int(first.img_pos.shape[0]) :]
    assert int(second_img_pos.min()) >= first.seq_len
    row_timesteps = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
    torch.testing.assert_close(row_timesteps[first.seq_len], torch.tensor(0.5))


def test_batched_forward_matches_per_request_forwards():
    """Packing two requests must not leak information across the boundary."""
    from vllm_omni.diffusion.models.minimax_h3.step_batch import minimax_h3_batched_forward_kwargs

    model = _SegmentMeanModel()
    first, first_video, first_audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=3)
    second, second_video, second_audio = _make_branch(text_len=5, latent_t=3, latent_h=6, latent_w=4, audio_t=2, seed=4)
    branches = [first, second]
    videos = [first_video, second_video]
    audios = [first_audio, second_audio]
    t_video = [0.25, 0.5]
    t_audio = [0.4, 0.6]

    batched_video, batched_audio = model(
        **minimax_h3_batched_forward_kwargs(
            branches=branches,
            video_rows=videos,
            audio_rows=audios,
            t_video=t_video,
            t_audio=t_audio,
            imgvid_cond_timesteps=[0.999, 0.999],
            audio_ref_cond_timesteps=[1.0, 1.0],
        )
    )
    video_parts = torch.split(batched_video, [int(branch.img_pos.shape[0]) for branch in branches])
    audio_parts = torch.split(batched_audio, [int(branch.audio_pos.shape[0]) for branch in branches])

    for index, branch in enumerate(branches):
        alone_video, alone_audio = model(
            **branch.forward_kwargs(
                video_rows=videos[index],
                audio_rows=audios[index],
                t_video=t_video[index],
                t_audio=t_audio[index],
                imgvid_cond_timestep=0.999,
                audio_ref_cond_timestep=1.0,
            )
        )
        torch.testing.assert_close(video_parts[index], alone_video)
        torch.testing.assert_close(audio_parts[index], alone_audio)


@pytest.mark.parametrize("packed_batch_supported", [True, False])
def test_step_execution_matches_request_mode_denoise_loop(packed_batch_supported):
    """Stepping through the contract must reproduce the request-mode loop."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop

    model = _SegmentMeanModel()
    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=5)
    sigmas_video = _sigmas(6, 12.0)
    sigmas_audio = _sigmas(6, 3.0)

    reference_video, reference_audio = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video_rows,
        initial_audio_rows=audio_rows,
        keyframe_cond_rows=None,
        sigmas_video=sigmas_video,
        sigmas_audio=sigmas_audio,
        device=torch.device("cpu"),
    )

    pipeline = _step_pipeline(model, packed_batch_supported=packed_batch_supported)
    state = _make_state("req-0", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
    input_batch = SimpleNamespace(states=(state,))

    steps = 0
    while not state.denoise_completed:
        noise_pred = pipeline.denoise_step(input_batch, states=[state])
        pipeline.step_scheduler(state, noise_pred)
        steps += 1

    assert steps == len(sigmas_video) - 1
    assert state.total_steps == steps
    torch.testing.assert_close(state.latents, reference_video)
    torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], reference_audio)


def test_batched_step_execution_matches_independent_requests():
    """Two co-batched requests must land where they would have landed alone."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    model = _SegmentMeanModel()
    specs = [
        dict(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=6),
        dict(text_len=5, latent_t=3, latent_h=6, latent_w=4, audio_t=2, seed=7),
    ]
    # Different step counts, so the batch composition changes mid-flight.
    schedules = [(_sigmas(6, 12.0), _sigmas(6, 3.0)), (_sigmas(4, 12.0), _sigmas(4, 3.0))]

    pipeline = _step_pipeline(model)

    alone: list[tuple[torch.Tensor, torch.Tensor]] = []
    for spec, (sigmas_video, sigmas_audio) in zip(specs, schedules):
        branch, video_rows, audio_rows = _make_branch(**spec)
        state = _make_state("solo", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
        while not state.denoise_completed:
            pipeline.step_scheduler(state, pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state]))
        alone.append((state.latents, state.extra[mod._STEP_AUDIO_ROWS]))

    states = []
    for index, (spec, (sigmas_video, sigmas_audio)) in enumerate(zip(specs, schedules)):
        branch, video_rows, audio_rows = _make_branch(**spec)
        states.append(_make_state(f"req-{index}", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio))

    active = list(states)
    while active:
        noise_pred = pipeline.denoise_step(SimpleNamespace(states=tuple(active)), states=active)
        offset = 0
        for state in active:
            rows = state.latents.shape[0]
            pipeline.step_scheduler(state, noise_pred[offset : offset + rows])
            offset += rows
        assert offset == noise_pred.shape[0]
        # Finished requests leave the batch, exactly like the runner drops them.
        active = [state for state in active if not state.denoise_completed]

    for state, (expected_video, expected_audio) in zip(states, alone):
        torch.testing.assert_close(state.latents, expected_video)
        torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], expected_audio)


def test_both_modes_publish_denoise_progress_for_gated_attention():
    """TRTLLM's skip gate and RAINFUSION's warmup stay dense without these."""
    from vllm_omni.diffusion import forward_context as fc
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop

    model = _SegmentMeanModel()
    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=14)
    sigmas_video = _sigmas(4, 12.0)
    sigmas_audio = _sigmas(4, 3.0)

    published: list[tuple[int | None, float | None]] = []

    class _Recorder:
        def __init__(self):
            self.denoise_step_idx = None
            self.denoise_timestep = None

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "denoise_timestep":
                published.append((self.denoise_step_idx, value))

    recorder = _Recorder()
    published.clear()  # drop the pair __init__ recorded
    original = fc._forward_context
    fc._forward_context = recorder
    try:
        minimax_h3_denoise_loop(
            model=model,
            positive=branch,
            initial_video_rows=video_rows,
            initial_audio_rows=audio_rows,
            keyframe_cond_rows=None,
            sigmas_video=sigmas_video,
            sigmas_audio=sigmas_audio,
            device=torch.device("cpu"),
        )
        request_mode = list(published)

        published.clear()
        pipeline = _step_pipeline(model)
        state = _make_state("req-0", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
        while not state.denoise_completed:
            pipeline.step_scheduler(state, pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state]))
        step_mode = list(published)
    finally:
        fc._forward_context = original

    num_steps = len(sigmas_video) - 1
    expected = [(step, sigmas_video[step]) for step in range(num_steps)]
    assert request_mode == expected + [(None, None)]
    # Step mode has no loop to close, so it publishes only the per-step pairs.
    assert step_mode == expected


def test_mixed_step_batch_leaves_gated_attention_dense():
    """A batch spanning different steps has no single progress point to publish."""
    from vllm_omni.diffusion import forward_context as fc

    model = _SegmentMeanModel()
    first, first_video, first_audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=15)
    second, second_video, second_audio = _make_branch(
        text_len=5, latent_t=3, latent_h=6, latent_w=4, audio_t=2, seed=16
    )
    sigmas_video, sigmas_audio = _sigmas(6, 12.0), _sigmas(6, 3.0)
    states = [
        _make_state("req-0", model, first, first_video, first_audio, sigmas_video, sigmas_audio),
        _make_state("req-1", model, second, second_video, second_audio, sigmas_video, sigmas_audio),
    ]
    states[1].step_index = 2  # admitted later, so it trails the batch

    recorder = SimpleNamespace(denoise_step_idx="unset", denoise_timestep="unset")
    original = fc._forward_context
    fc._forward_context = recorder
    try:
        _step_pipeline(model).denoise_step(SimpleNamespace(states=tuple(states)), states=states)
    finally:
        fc._forward_context = original

    assert recorder.denoise_step_idx is None
    assert recorder.denoise_timestep is None


def test_abort_stops_h3_after_the_inflight_step():
    """H3 relies on the generic step scheduler to stop aborted requests."""
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.sched import StepScheduler
    from vllm_omni.diffusion.worker.utils import RunnerOutput
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=9)
    model = _SegmentMeanModel()
    state = _make_state("req-abort", model, branch, video_rows, audio_rows, _sigmas(6, 12.0), _sigmas(6, 3.0))
    pipeline = _step_pipeline(model)
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace())

    engine = object.__new__(DiffusionEngine)
    engine.scheduler = scheduler
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._closed = False
    engine.abort_queue = queue.Queue()
    calls = 0

    def execute_h3_step(scheduler_output):
        nonlocal calls
        calls += 1
        assert scheduler_output.scheduled_request_ids == ["req-abort"]
        noise_pred = pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state])
        pipeline.step_scheduler(state, noise_pred)
        engine.abort("req-abort")
        return RunnerOutput(
            request_id=state.request_id,
            step_index=state.step_index,
            finished=False,
            result=None,
        )

    engine.execute_fn = execute_h3_step
    output = engine.add_req_and_wait_for_response(
        OmniDiffusionRequest(
            prompt="abort H3",
            request_id="req-abort",
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=state.total_steps),
        )
    )

    assert calls == 1
    assert state.step_index == 1
    assert output == DiffusionOutput(aborted=True, abort_message="Request req-abort aborted.")


def test_prepare_encode_seeds_runner_visible_state(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=8)
    sigmas_video = _sigmas(6, 12.0)
    sigmas_audio = _sigmas(6, 3.0)
    context = {
        "height": 96,
        "width": 64,
        "latent_t": 2,
        "latent_h": 4,
        "latent_w": 6,
        "audio_t": 3,
        **{key: None for key in mod._MINIMAX_H3_DENOISE_INPUT_KEYS},
    }

    pipeline = _step_pipeline(_SegmentMeanModel())
    monkeypatch.setattr(mod.MiniMaxH3Pipeline, "_extract_prompt", staticmethod(lambda _: ("a prompt", {})))
    monkeypatch.setattr(mod.MiniMaxH3Pipeline, "_prepare_request_inputs", lambda self, **_: context)
    monkeypatch.setattr(
        mod.MiniMaxH3Pipeline,
        "_build_denoise_inputs",
        lambda self, **_: {
            "branch": branch,
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "cond_anchor": None,
            "audio_anchor": None,
            "sigmas_video": sigmas_video,
            "sigmas_audio": sigmas_audio,
        },
    )

    from vllm_omni.diffusion.worker.utils import StepRequestState

    state = StepRequestState(
        request_id="req-0",
        sampling=SimpleNamespace(num_outputs_per_prompt=1),
        prompt="a prompt",
    )
    pipeline.prepare_encode(state)

    # The runner slices the batched velocity by this row count.
    assert state.latents.shape == (int(branch.img_pos.shape[0]), 96)
    assert state.total_steps == len(sigmas_video) - 1
    assert state.step_index == 0
    assert state.do_true_cfg is False
    torch.testing.assert_close(state.current_timestep, torch.tensor(1.0 - sigmas_video[0]))
    assert state.extra[mod._STEP_BRANCH] is branch
    assert state.extra[mod._STEP_SHAPE]["height"] == 96


def test_prepare_encode_rejects_request_mode_only_features():
    """Multi-output and DLO have no representation in the step contract."""
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.errors import OmniClientError

    # A request state carries exactly one latent tensor.
    multi_output = StepRequestState(
        request_id="req-0",
        sampling=SimpleNamespace(num_outputs_per_prompt=2),
        prompt="a prompt",
    )
    with pytest.raises(OmniClientError, match="one output per request"):
        _step_pipeline(_SegmentMeanModel()).prepare_encode(multi_output)

    # Distributed layerwise offload streams the DiT around a whole denoise loop.
    single_output = StepRequestState(
        request_id="req-1",
        sampling=SimpleNamespace(num_outputs_per_prompt=1),
        prompt="a prompt",
    )
    dlo_pipeline = _step_pipeline(_SegmentMeanModel())
    dlo_pipeline._dlo_residency_controller = object()
    with pytest.raises(ValueError, match="distributed layerwise offload"):
        dlo_pipeline.prepare_encode(single_output)


def _batched_kwargs(branches, videos, audios):
    from vllm_omni.diffusion.models.minimax_h3.step_batch import minimax_h3_batched_forward_kwargs

    return minimax_h3_batched_forward_kwargs(
        branches=branches,
        video_rows=videos,
        audio_rows=audios,
        t_video=[0.25] * len(branches),
        t_audio=[0.4] * len(branches),
        imgvid_cond_timesteps=[0.999] * len(branches),
        audio_ref_cond_timesteps=[1.0] * len(branches),
    )


def test_packed_batch_reports_its_request_count():
    """Attention needs the request count; cu_seqlens length cannot supply it.

    A request whose rows already land on the 64-row alignment contributes an
    empty padding document, so bound counts alone cannot tell two such requests
    apart from one padded request.
    """
    first, first_video, first_audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=9)
    second, second_video, second_audio = _make_branch(
        text_len=5, latent_t=3, latent_h=6, latent_w=4, audio_t=2, seed=10
    )
    # text 46 + audio 4 + video 14 = 64 rows exactly: no padding tail.
    aligned, aligned_video, aligned_audio = _make_branch(
        text_len=46, latent_t=7, latent_h=2, latent_w=4, audio_t=2, seed=11
    )
    assert aligned.used_len == aligned.seq_len

    solo = _batched_kwargs([first], [first_video], [first_audio])["packed_seq_params"]
    assert solo["num_requests"] == 1

    batched = _batched_kwargs(
        [first, second],
        [first_video, second_video],
        [first_audio, second_audio],
    )["packed_seq_params"]
    assert batched["num_requests"] == 2

    # Two alignment-exact requests: five bounds, two of them empty documents.
    aligned_pair = _batched_kwargs(
        [aligned, aligned],
        [aligned_video, aligned_video],
        [aligned_audio, aligned_audio],
    )["packed_seq_params"]
    assert aligned_pair["num_requests"] == 2
    assert aligned_pair["cu_seqlens_q"].tolist() == [
        0,
        aligned.seq_len,
        aligned.seq_len,
        2 * aligned.seq_len,
        2 * aligned.seq_len,
    ]


def test_alignment_exact_requests_stay_isolated():
    """The no-padding-tail layout must not leak across the request boundary."""
    model = _SegmentMeanModel()
    branch, video_rows, audio_rows = _make_branch(text_len=46, latent_t=7, latent_h=2, latent_w=4, audio_t=2, seed=12)
    other, other_video, other_audio = _make_branch(text_len=46, latent_t=7, latent_h=2, latent_w=4, audio_t=2, seed=13)
    assert branch.used_len == branch.seq_len

    batched_video, batched_audio = model(
        **_batched_kwargs([branch, other], [video_rows, other_video], [audio_rows, other_audio])
    )
    video_parts = torch.split(batched_video, [int(b.img_pos.shape[0]) for b in (branch, other)])
    audio_parts = torch.split(batched_audio, [int(b.audio_pos.shape[0]) for b in (branch, other)])

    for index, (one, rows, audio) in enumerate([(branch, video_rows, audio_rows), (other, other_video, other_audio)]):
        alone_video, alone_audio = model(
            **one.forward_kwargs(
                video_rows=rows,
                audio_rows=audio,
                t_video=0.25,
                t_audio=0.4,
                imgvid_cond_timestep=0.999,
                audio_ref_cond_timestep=1.0,
            )
        )
        torch.testing.assert_close(video_parts[index], alone_video)
        torch.testing.assert_close(audio_parts[index], alone_audio)
