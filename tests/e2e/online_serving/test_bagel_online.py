# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end online serving test for Bagel text2img and img2img generation.

This test validates that the Bagel model can serve image generation requests
via the OpenAI-compatible chat completions API.

Equivalent to running:
    vllm-omni serve "ByteDance-Seed/BAGEL-7B-MoT" --omni --port 8091

    # text2img
    python3 examples/online_serving/bagel/openai_chat_client.py \\
        --prompt "A cute cat" --modality text2img

    # img2img
    python3 examples/online_serving/bagel/openai_chat_client.py \\
        --prompt "Let the woman wear a blue dress" --modality img2img \\
        --image-url women.jpg
"""

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.conftest import OmniServerParams
from tests.e2e.offline_inference.test_bagel_img2img import DEFAULT_PROMPT as IMG2IMG_PROMPT
from tests.e2e.offline_inference.test_bagel_img2img import REFERENCE_PIXELS as IMG2IMG_REFERENCE_PIXELS
from tests.e2e.offline_inference.test_bagel_text2img import DEFAULT_PROMPT as TEXT2IMG_PROMPT
from tests.e2e.offline_inference.test_bagel_text2img import REFERENCE_PIXELS as TEXT2IMG_REFERENCE_PIXELS
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
STAGE_CONFIGS_PATH = str(
    Path(__file__).parent.parent / "offline_inference" / "stage_configs" / "bagel_sharedmemory_ci.yaml"
)

PIXEL_TOLERANCE = 5

# Create parameter combinations for model and stage config
test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIGS_PATH,
        server_args=["--stage-init-timeout", "300"],
    ),
]


def _build_text2img_messages(prompt: str) -> list[dict]:
    """Build OpenAI-format messages for text2img generation."""
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"}],
        }
    ]


def _build_img2img_messages(prompt: str, image_b64: str) -> list[dict]:
    """Build OpenAI-format messages for img2img generation."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]


def _validate_pixels(
    image: Image.Image,
    reference_pixels: list[dict[str, Any]],
    tolerance: int = PIXEL_TOLERANCE,
) -> None:
    """Validate that image pixels match expected reference values."""
    for ref in reference_pixels:
        x, y = ref["position"]
        expected = ref["rgb"]
        actual = image.getpixel((x, y))[:3]
        assert all(abs(a - e) <= tolerance for a, e in zip(actual, expected)), (
            f"Pixel mismatch at ({x}, {y}): expected {expected}, got {actual}"
        )


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_bagel_text2img_online(omni_server, openai_client) -> None:
    """Test Bagel text2img via OpenAI-compatible chat completions API."""
    request_config = {
        "model": omni_server.model,
        "messages": _build_text2img_messages(TEXT2IMG_PROMPT),
        "modalities": ["image"],
    }

    responses = openai_client.send_diffusion_request(request_config)
    assert responses, "No responses received"

    image = responses[0].images[0]
    assert image is not None, "No image in response"

    w, h = image.size
    assert (w, h) == (1024, 1024), f"Expected 1024x1024, got {image.size}"
    _validate_pixels(image, TEXT2IMG_REFERENCE_PIXELS)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_bagel_img2img_online(omni_server, openai_client) -> None:
    """Test Bagel img2img via OpenAI-compatible chat completions API."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    buffer = BytesIO()
    input_image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    request_config = {
        "model": omni_server.model,
        "messages": _build_img2img_messages(IMG2IMG_PROMPT, image_b64),
        "modalities": ["image"],
    }

    responses = openai_client.send_diffusion_request(request_config)
    assert responses, "No responses received"

    image = responses[0].images[0]
    assert image is not None, "No image in response"

    w, h = image.size
    assert (w, h) == (1024, 672), f"Expected 1024x672, got {image.size}"
    _validate_pixels(image, IMG2IMG_REFERENCE_PIXELS)
