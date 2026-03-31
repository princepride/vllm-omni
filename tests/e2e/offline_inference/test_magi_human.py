# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for MagiHuman pipeline via vLLM-Omni."""

import numpy as np
import pytest

from tests.utils import hardware_test
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _validate_video_quality(video_np: np.ndarray) -> None:
    """Validate that the video contains meaningful content (not black/corrupt).

    Diffusion outputs vary across TP sizes and hardware, so we check
    statistical properties rather than exact pixel values.
    """
    assert video_np.dtype == np.uint8, f"Expected uint8, got {video_np.dtype}"

    first_frame = video_np[0].astype(np.float32)

    mean_val = first_frame.mean()
    assert mean_val > 30, f"Video appears mostly black: mean pixel value {mean_val:.1f} (expected > 30)"
    assert mean_val < 240, f"Video appears mostly white: mean pixel value {mean_val:.1f} (expected < 240)"

    std_val = first_frame.std()
    assert std_val > 10, f"Video has near-zero variance (solid color): std {std_val:.1f} (expected > 10)"

    non_zero_ratio = np.count_nonzero(first_frame) / first_frame.size
    assert non_zero_ratio > 0.5, f"Too many zero pixels: {non_zero_ratio:.2%} non-zero (expected > 50%)"

    assert not np.any(np.isnan(first_frame)), "Video contains NaN values"


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_magi_human_e2e(run_level):
    """End-to-end test for MagiHuman generating video and audio."""
    if run_level != "advanced_model":
        pytest.skip("MagiHuman e2e test requires advanced_model run level with real weights.")

    model_path = "princepride/daVinci-MagiHuman"

    omni = Omni(
        model=model_path,
        init_timeout=1200,
        tensor_parallel_size=2,
    )

    prompt = (
        "A young woman with long, wavy golden blonde hair and bright blue eyes, "
        "wearing a fitted ivory silk blouse with a delicate lace collar, sits "
        "stationary in front of a softly lit, blurred warm-toned interior. Her "
        "overall disposition is warm, composed, and gently confident. The camera "
        "holds a static medium close-up, framing her from the shoulders up, "
        "with shallow depth of field keeping her face in sharp focus. Soft "
        "directional key light falls from the upper left, casting a gentle "
        "highlight along her cheekbone and nose bridge. She draws a quiet breath, "
        "the levator labii superiors relaxing as her lips part. She speaks in "
        "clear, warm, unhurried American English: "
        "\"The most beautiful things in life aren't things at all — "
        "they're moments, feelings, and the people who make you feel truly alive.\" "
        "Her jaw descends smoothly on each stressed syllable; the orbicularis oris "
        "shapes each vowel with precision. A faint, genuine smile engages the "
        "zygomaticus major, lifting her lip corners fractionally. Her brows rest "
        "in a soft, neutral arch throughout. She maintains steady, forward-facing "
        "eye contact. Head position remains level; no torso displacement occurs.\n\n"
        "Dialogue:\n"
        "<Young blonde woman, American English>: "
        "\"The most beautiful things in life aren't things at all — "
        "they're moments, feelings, and the people who make you feel truly alive.\"\n\n"
        "Background Sound:\n"
        "<Soft, warm indoor ambience with a faint distant piano melody>"
    )

    sampling_params = OmniDiffusionSamplingParams(
        height=256,
        width=448,
        num_inference_steps=8,
        seed=52,
        extra_args={
            "seconds": 5,
            "sr_height": 1080,
            "sr_width": 1920,
            "sr_num_inference_steps": 5,
        },
    )

    try:
        outputs = list(
            omni.generate(
                prompts=[prompt],
                sampling_params_list=[sampling_params],
            )
        )

        assert len(outputs) > 0, "No outputs returned"
        first = outputs[0]
        req_out = first.request_output
        assert hasattr(req_out, "custom_output") and req_out.custom_output, "No custom_output found"

        custom = req_out.custom_output
        assert "video" in custom and custom["video"] is not None, "No video generated"
        assert "audio" in custom and custom["audio"] is not None, "No audio generated"

        video_np = custom["video"]
        assert video_np.shape[1] in (1056, 1080) and video_np.shape[2] == 1920, (
            f"Unexpected video shape: {video_np.shape}"
        )

        _validate_video_quality(video_np)
    finally:
        omni.close()
