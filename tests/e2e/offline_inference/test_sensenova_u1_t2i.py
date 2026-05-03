# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for SenseNova-U1 text2img generation.

This test validates that the SenseNova-U1 model generates images that match
expected reference pixel values within a ±10 tolerance.

Equivalent to running:
    python SenseNova-U1/examples/t2i/inference.py \
        --model_path SenseNova/SenseNova-U1-8B-MoT \
        --prompt "Close portrait of an elderly woman ..." \
        --width 1536 --height 2720 \
        --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 \
        --num_steps 50 --seed 42 --think
"""

import os
from typing import Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.omni import Omni

SENSENOVA_U1_CI_DEPLOY = get_deploy_config_path("ci/sensenova_u1.yaml")

# Reference pixel data extracted from the known-good output image
# Each entry contains (x, y) position and expected (R, G, B) values
# Generated with seed=42, num_inference_steps=50, think=True,
# prompt="Close portrait of an elderly woman by a farmhouse window, ..."
REFERENCE_PIXELS = [
    {"position": (100, 100), "rgb": (247, 249, 250)},
    {"position": (768, 200), "rgb": (176, 135, 97)},
    {"position": (400, 600), "rgb": (200, 190, 180)},
    {"position": (1200, 2000), "rgb": (92, 77, 65)},
    {"position": (750, 500), "rgb": (186, 135, 88)},
    {"position": (300, 1360), "rgb": (198, 159, 114)},
    {"position": (1000, 1800), "rgb": (57, 29, 13)},
    {"position": (500, 2400), "rgb": (97, 83, 74)},
    {"position": (768, 1360), "rgb": (84, 42, 22)},
    {"position": (200, 900), "rgb": (195, 190, 182)},
]

PIXEL_TOLERANCE = 10

DEFAULT_PROMPT = (
    "Close portrait of an elderly woman by a farmhouse window, textured skin, "
    "gentle smile, warm natural light, emotional documentary look. The portrait "
    "should feel polished and natural, with sharp eyes, realistic skin texture, "
    "accurate facial anatomy, and premium lighting that keeps the face as the "
    "main focus."
)

EXPECTED_OUTPUT_SIZE = (1536, 2720)


def _configure_sampling_params(omni: Omni) -> list:
    """Configure sampling parameters for SenseNova-U1 text2img generation."""
    params_list = omni.default_sampling_params_list
    if len(params_list) > 0:
        params_list[0].num_inference_steps = 50  # type: ignore
        params_list[0].height = EXPECTED_OUTPUT_SIZE[1]  # type: ignore
        params_list[0].width = EXPECTED_OUTPUT_SIZE[0]  # type: ignore
        params_list[0].extra_args = {  # type: ignore
            "cfg_scale": 4.0,
            "cfg_norm": "none",
            "timestep_shift": 3.0,
            "cfg_interval": (0.0, 1.0),
            "batch_size": 1,
            "think": True,
            "t_eps": 0.02,
        }
    return params_list


def _extract_generated_image(omni_outputs: list) -> Image.Image | None:
    """Extract the generated image from Omni outputs."""
    for req_output in omni_outputs:
        if images := getattr(req_output, "images", None):
            return images[0]
        if hasattr(req_output, "request_output") and req_output.request_output:
            stage_out = req_output.request_output
            if hasattr(stage_out, "images") and stage_out.images:
                return stage_out.images[0]
    return None


def _validate_pixels(
    image: Image.Image,
    reference_pixels: list[dict[str, Any]] = REFERENCE_PIXELS,
    tolerance: int = PIXEL_TOLERANCE,
) -> None:
    """Validate that image pixels match expected reference values.

    Args:
        image: The PIL Image to validate.
        reference_pixels: List of dicts with 'position' (x, y) and 'rgb' (R, G, B).
        tolerance: Maximum allowed difference per color channel.

    Raises:
        AssertionError: If any pixel differs beyond tolerance.
    """
    for ref in reference_pixels:
        x, y = ref["position"]
        expected = ref["rgb"]
        actual = image.getpixel((x, y))[:3]
        assert all(abs(a - e) <= tolerance for a, e in zip(actual, expected)), (
            f"Pixel mismatch at ({x}, {y}): expected {expected}, got {actual}"
        )


def _generate_sensenova_u1_image(
    omni: Omni,
    prompt: str = DEFAULT_PROMPT,
) -> Image.Image:
    """Generate an image using SenseNova-U1 model with configured parameters.

    Args:
        omni: The Omni instance to use for generation.
        prompt: The text prompt for image generation.

    Returns:
        The generated PIL Image.

    Raises:
        AssertionError: If no image is generated or size is incorrect.
    """
    params_list = _configure_sampling_params(omni)

    omni_outputs = list(
        omni.generate(
            prompts=[{"prompt": prompt, "modalities": ["image"]}],
            sampling_params_list=params_list,
        )
    )

    generated_image = _extract_generated_image(omni_outputs)
    assert generated_image is not None, "No images generated"
    assert generated_image.size == EXPECTED_OUTPUT_SIZE, f"Expected {EXPECTED_OUTPUT_SIZE}, got {generated_image.size}"

    return generated_image


def _resolve_deploy_config(config_path: str, run_level: str) -> str:
    """Resolve deploy config based on run level.

    For advanced_model (real weights), strip load_format: dummy so the model
    falls back to loading real weights from HuggingFace.
    """
    if run_level == "advanced_model":
        return modify_stage_config(
            config_path,
            deletes={
                "stages": {
                    0: ["load_format"],
                }
            },
        )
    return config_path


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_sensenova_u1_text2img(run_level):
    """Test SenseNova-U1 text2img with shared memory connector."""
    config_path = _resolve_deploy_config(SENSENOVA_U1_CI_DEPLOY, run_level)
    with OmniRunner(
        "SenseNova/SenseNova-U1-8B-MoT",
        stage_configs_path=config_path,
    ) as runner:
        generated_image = _generate_sensenova_u1_image(runner.omni)
        if run_level == "advanced_model":
            _validate_pixels(generated_image)
