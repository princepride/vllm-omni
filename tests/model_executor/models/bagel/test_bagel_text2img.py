# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for Bagel text2img generation.

This test validates that the Bagel model generates images that match
expected reference pixel values within a ±5 tolerance.

Equivalent to running:
    python3 examples/offline_inference/bagel/end2end.py \
        --prompts "A futuristic city skyline at twilight, cyberpunk style" \
        --modality text2img --step 15
"""

# Reference pixel data extracted from the known-good output image
# Each entry contains (x, y) position and expected (R, G, B) values
REFERENCE_PIXELS = [
    {"position": (100, 100), "rgb": (68, 107, 134)},
    {"position": (400, 50), "rgb": (95, 139, 166)},
    {"position": (700, 100), "rgb": (99, 122, 151)},
    {"position": (150, 400), "rgb": (111, 125, 153)},
    {"position": (512, 512), "rgb": (97, 107, 131)},
    {"position": (700, 400), "rgb": (48, 64, 98)},
    {"position": (100, 700), "rgb": (79, 63, 84)},
    {"position": (400, 700), "rgb": (40, 58, 79)},
    {"position": (700, 700), "rgb": (60, 75, 103)},
    {"position": (256, 256), "rgb": (97, 128, 156)},
]

# Maximum allowed difference per color channel
PIXEL_TOLERANCE = 5


def test_bagel_text2img_pixel_consistency():
    """Test that Bagel text2img generates images with consistent pixel values.

    This end-to-end test:
    1. Initializes the Omni client with the BAGEL-7B-MoT model
    2. Generates an image using the test prompt
    3. Compares 10 reference pixel values against the generated image
    4. Asserts each RGB channel differs by at most ±5
    """
    from vllm_omni.entrypoints.omni import Omni

    # Initialize the Omni client
    model_name = "ByteDance-Seed/BAGEL-7B-MoT"
    omni = Omni(model=model_name)

    try:
        # Prepare the prompt (text2img format from end2end.py)
        prompt_text = "A futuristic city skyline at twilight, cyberpunk style"
        formatted_prompt = {
            "prompt": f"<|im_start|>{prompt_text}<|im_end|>",
            "modalities": ["image"],
        }

        # Get default sampling params and configure for text2img with 15 steps
        params_list = omni.default_sampling_params_list
        # Set max_tokens=1 for the first stage (LLM stage)
        params_list[0].max_tokens = 1  # type: ignore
        # Set num_inference_steps=15 for the diffusion stage
        if len(params_list) > 1:
            params_list[1].num_inference_steps = 15  # type: ignore

        # Generate the image
        omni_outputs = list(omni.generate(prompts=[formatted_prompt], sampling_params_list=params_list))

        # Extract the generated image from OmniRequestOutput
        # Following the exact extraction logic from end2end.py
        assert len(omni_outputs) > 0, "Expected at least one output"

        generated_image = None
        for req_output in omni_outputs:
            # First check if images are directly on the output
            images = getattr(req_output, "images", None)
            if images and len(images) > 0:
                generated_image = images[0]
                break

            # Then check output attribute (for diffusion direct outputs)
            if not images and hasattr(req_output, "output"):
                if isinstance(req_output.output, list):
                    images = req_output.output
                else:
                    images = [req_output.output] if req_output.output else None
                if images and len(images) > 0:
                    from PIL import Image

                    if isinstance(images[0], Image.Image):
                        generated_image = images[0]
                        break

            # Finally check request_output for stage outputs (pipeline mode)
            if hasattr(req_output, "request_output") and req_output.request_output:
                request_output = req_output.request_output
                # request_output might be iterable (list of stage outputs) or single object
                if hasattr(request_output, "__iter__") and not isinstance(request_output, (str, bytes)):
                    for stage_out in request_output:
                        if hasattr(stage_out, "images") and stage_out.images:
                            generated_image = stage_out.images[0]
                            break
                elif hasattr(request_output, "images") and request_output.images:
                    generated_image = request_output.images[0]
                if generated_image:
                    break

        assert generated_image is not None, f"No images generated. Outputs: {omni_outputs}"

        # Verify image dimensions
        width, height = generated_image.size
        assert width == 1024 and height == 1024, f"Expected 1024x1024 image, got {width}x{height}"

        # Compare reference pixels
        mismatches = []
        for ref in REFERENCE_PIXELS:
            x, y = ref["position"]
            expected_r, expected_g, expected_b = ref["rgb"]

            # Get actual pixel value
            actual_pixel = generated_image.getpixel((x, y))
            actual_r, actual_g, actual_b = actual_pixel[0], actual_pixel[1], actual_pixel[2]

            # Check each channel
            diff_r = abs(actual_r - expected_r)
            diff_g = abs(actual_g - expected_g)
            diff_b = abs(actual_b - expected_b)

            if diff_r > PIXEL_TOLERANCE or diff_g > PIXEL_TOLERANCE or diff_b > PIXEL_TOLERANCE:
                mismatches.append(
                    f"Position ({x}, {y}): "
                    f"expected RGB({expected_r}, {expected_g}, {expected_b}), "
                    f"got RGB({actual_r}, {actual_g}, {actual_b}), "
                    f"diff: ({diff_r}, {diff_g}, {diff_b})"
                )

        # Assert no mismatches
        assert len(mismatches) == 0, f"Pixel value mismatches (tolerance=±{PIXEL_TOLERANCE}):\n" + "\n".join(mismatches)

    finally:
        # Clean up
        omni.close()
