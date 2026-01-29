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

import os
import signal
import socket
import subprocess
import tempfile
import time

import yaml

from vllm_omni.entrypoints.omni import Omni

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


def test_bagel_text2img_shared_memory_connector():
    """Test Bagel text2img with shared memory connector."""

    omni = Omni(model="ByteDance-Seed/BAGEL-7B-MoT")

    try:
        params_list = omni.default_sampling_params_list
        params_list[0].max_tokens = 1  # type: ignore
        if len(params_list) > 1:
            params_list[1].num_inference_steps = 15  # type: ignore

        omni_outputs = list(
            omni.generate(
                prompts=[
                    {
                        "prompt": "<|im_start|>A futuristic city skyline at twilight, cyberpunk style<|im_end|>",
                        "modalities": ["image"],
                    }
                ],
                sampling_params_list=params_list,
            )
        )

        # Extract generated image
        generated_image = None
        for req_output in omni_outputs:
            if images := getattr(req_output, "images", None):
                generated_image = images[0]
                break
            if hasattr(req_output, "request_output") and req_output.request_output:
                for stage_out in req_output.request_output:
                    if hasattr(stage_out, "images") and stage_out.images:
                        generated_image = stage_out.images[0]
                        break
                if generated_image:
                    break

        assert generated_image is not None, "No images generated"
        assert generated_image.size == (1024, 1024), f"Expected 1024x1024, got {generated_image.size}"

        # Validate pixels
        for ref in REFERENCE_PIXELS:
            x, y = ref["position"]
            expected = ref["rgb"]
            actual = generated_image.getpixel((x, y))[:3]
            assert all(abs(a - e) <= PIXEL_TOLERANCE for a, e in zip(actual, expected)), f"Pixel mismatch at ({x}, {y})"

    finally:
        omni.close()


def test_bagel_text2img_mooncake_connector():
    """Test Bagel text2img with Mooncake connector for inter-stage communication."""

    MOONCAKE_HOST = "127.0.0.1"
    MOONCAKE_RPC_PORT = 50051
    MOONCAKE_HTTP_PORT = 8080
    MOONCAKE_METRICS_PORT = 9003

    def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
        """Wait for a port to become available."""
        for _ in range(timeout):
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                time.sleep(1)
        return False

    # Stage configuration with Mooncake connector
    mooncake_config = {
        "stage_args": [
            {
                "stage_id": 0,
                "stage_type": "llm",
                "runtime": {"devices": "0", "max_batch_size": 1},
                "engine_args": {
                    "model_stage": "thinker",
                    "model_arch": "BagelForConditionalGeneration",
                    "worker_type": "ar",
                    "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
                    "gpu_memory_utilization": 0.35,
                    "enforce_eager": True,
                    "trust_remote_code": True,
                    "engine_output_type": "text",
                    "distributed_executor_backend": "mp",
                    "enable_prefix_caching": False,
                    "max_num_batched_tokens": 32768,
                    "tensor_parallel_size": 1,
                    "omni_kv_config": {"need_send_cache": True, "kv_transfer_criteria": {"type": "prefill_finished"}},
                },
                "final_output": True,
                "final_output_type": "text",
                "is_comprehension": True,
                "default_sampling_params": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "top_k": 1,
                    "max_tokens": 2048,
                    "seed": 52,
                    "detokenize": True,
                    "repetition_penalty": 1.05,
                },
                "output_connectors": {"to_stage_1": "mooncake_connector"},
            },
            {
                "stage_id": 1,
                "stage_type": "diffusion",
                "runtime": {"devices": "0", "max_batch_size": 1},
                "engine_args": {
                    "model_stage": "dit",
                    "gpu_memory_utilization": 0.55,
                    "enforce_eager": True,
                    "trust_remote_code": True,
                    "engine_output_type": "image",
                    "distributed_executor_backend": "mp",
                    "enable_prefix_caching": False,
                    "max_num_batched_tokens": 32768,
                    "tensor_parallel_size": 1,
                    "omni_kv_config": {"need_recv_cache": True},
                },
                "engine_input_source": [0],
                "final_output": True,
                "final_output_type": "image",
                "is_comprehension": False,
                "default_sampling_params": {"seed": 52},
                "input_connectors": {"from_stage_0": "mooncake_connector"},
            },
        ],
        "runtime": {
            "enabled": True,
            "defaults": {"window_size": -1, "max_inflight": 1},
            "connectors": {
                "mooncake_connector": {
                    "name": "MooncakeConnector",
                    "extra": {
                        "host": MOONCAKE_HOST,
                        "metadata_server": f"http://{MOONCAKE_HOST}:{MOONCAKE_HTTP_PORT}/metadata",
                        "master": f"{MOONCAKE_HOST}:{MOONCAKE_RPC_PORT}",
                        "segment": 512000000,
                        "localbuf": 64000000,
                        "proto": "tcp",
                    },
                },
            },
            "edges": [{"from": 0, "to": 1, "window_size": -1}],
        },
    }

    mooncake_master_proc = None
    temp_config_file = None
    omni = None

    try:
        # Clean up existing mooncake_master processes
        subprocess.run(["pkill", "-9", "-f", "mooncake_master"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

        # Start mooncake_master
        mooncake_master_proc = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={MOONCAKE_RPC_PORT}",
                "--enable_http_metadata_server=true",
                "--http_metadata_server_host=0.0.0.0",
                f"--http_metadata_server_port={MOONCAKE_HTTP_PORT}",
                f"--metrics_port={MOONCAKE_METRICS_PORT}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        assert wait_for_port(MOONCAKE_HOST, MOONCAKE_RPC_PORT), "mooncake_master failed to start"

        # Create temp config and initialize Omni
        temp_config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(mooncake_config, temp_config_file)
        temp_config_file.close()

        omni = Omni(model="ByteDance-Seed/BAGEL-7B-MoT", stage_configs_path=temp_config_file.name)

        # Generate image
        params_list = omni.default_sampling_params_list
        params_list[0].max_tokens = 1  # type: ignore
        if len(params_list) > 1:
            params_list[1].num_inference_steps = 15  # type: ignore

        omni_outputs = list(
            omni.generate(
                prompts=[
                    {
                        "prompt": "<|im_start|>A futuristic city skyline at twilight, cyberpunk style<|im_end|>",
                        "modalities": ["image"],
                    }
                ],
                sampling_params_list=params_list,
            )
        )

        # Extract generated image
        generated_image = None
        for req_output in omni_outputs:
            if images := getattr(req_output, "images", None):
                generated_image = images[0]
                break
            if hasattr(req_output, "request_output") and req_output.request_output:
                for stage_out in req_output.request_output:
                    if hasattr(stage_out, "images") and stage_out.images:
                        generated_image = stage_out.images[0]
                        break
                if generated_image:
                    break

        assert generated_image is not None, "No images generated"
        assert generated_image.size == (1024, 1024), f"Expected 1024x1024, got {generated_image.size}"

        # Validate pixels
        for ref in REFERENCE_PIXELS:
            x, y = ref["position"]
            expected = ref["rgb"]
            actual = generated_image.getpixel((x, y))[:3]
            assert all(abs(a - e) <= PIXEL_TOLERANCE for a, e in zip(actual, expected)), f"Pixel mismatch at ({x}, {y})"

    finally:
        if omni:
            omni.close()
        if temp_config_file:
            try:
                os.unlink(temp_config_file.name)
            except OSError:
                pass
        if mooncake_master_proc:
            try:
                os.killpg(os.getpgid(mooncake_master_proc.pid), signal.SIGKILL)
            except OSError:
                pass
