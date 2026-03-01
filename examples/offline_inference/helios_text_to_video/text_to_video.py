# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Helios Text-to-Video generation example.

Usage (Helios-Base, Stage 1 only):
    python helios_text_to_video.py \
        --model /path/to/Helios-Base \
        --prompt "A serene lakeside sunrise with mist over the water." \
        --height 384 --width 640 --num-frames 33 \
        --num-inference-steps 30 --guidance-scale 5.0

Usage (Helios-Mid, Stage 2 + CFG-Zero*):
    python helios_text_to_video.py \
        --model /path/to/Helios-Mid \
        --prompt "A serene lakeside sunrise with mist over the water." \
        --height 384 --width 640 --num-frames 33 \
        --guidance-scale 5.0 \
        --is-enable-stage2 \
        --pyramid-num-inference-steps-list 20 20 20 \
        --use-cfg-zero-star --use-zero-init --zero-steps 1

Usage (Helios-Distilled, Stage 2 pyramid + DMD):
    python helios_text_to_video.py \
        --model /path/to/Helios-Distilled \
        --prompt "A serene lakeside sunrise with mist over the water." \
        --height 384 --width 640 --num-frames 33 \
        --guidance-scale 1.0 \
        --is-enable-stage2 \
        --pyramid-num-inference-steps-list 2 2 2 \
        --is-amplify-first-chunk
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with Helios T2V.")
    parser.add_argument(
        "--model",
        default="BestWishYsh/Helios-Base",
        help="Helios model ID or local path (e.g. Helios-Base, Helios-Distilled).",
    )
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.", help="Text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--height", type=int, default=384, help="Video height.")
    parser.add_argument("--width", type=int, default=640, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=33, help="Number of video frames.")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Sampling steps (Stage 1 only).")
    parser.add_argument("--output", type=str, default="helios_output.mp4", help="Output video path.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video.")

    # Stage 2 (pyramid multi-stage denoising)
    parser.add_argument(
        "--is-enable-stage2",
        action="store_true",
        help="Enable pyramid multi-stage denoising (Stage 2). Required for Helios-Distilled.",
    )
    parser.add_argument(
        "--pyramid-num-stages",
        type=int,
        default=3,
        help="Number of pyramid stages for Stage 2.",
    )
    parser.add_argument(
        "--pyramid-num-inference-steps-list",
        type=int,
        nargs="+",
        default=[10, 10, 10],
        help="Inference steps per pyramid stage.",
    )

    # DMD
    parser.add_argument(
        "--is-amplify-first-chunk",
        action="store_true",
        help="Enable DMD amplification for the first chunk (Helios-Distilled).",
    )

    # CFG Zero Star
    parser.add_argument(
        "--use-cfg-zero-star",
        action="store_true",
        help="Enable CFG Zero Star guidance (recommended for Helios-Mid).",
    )
    parser.add_argument(
        "--use-zero-init",
        action="store_true",
        help="Use zero initialization for the first denoising steps with CFG-Zero*.",
    )
    parser.add_argument(
        "--zero-steps",
        type=int,
        default=1,
        help="Number of initial denoising steps using zero prediction (default: 1).",
    )

    # Memory & parallelism
    parser.add_argument("--vae-use-slicing", action="store_true", help="Enable VAE slicing.")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offloading.")
    parser.add_argument("--enable-layerwise-offload", action="store_true", help="Enable layerwise offloading.")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses SP degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring SP degree.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism size.")

    return parser.parse_args()


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    omni = Omni(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
    )

    # Build extra_args for Helios-specific parameters
    extra_args = {}
    if args.is_enable_stage2:
        extra_args["is_enable_stage2"] = True
        extra_args["pyramid_num_stages"] = args.pyramid_num_stages
        extra_args["pyramid_num_inference_steps_list"] = args.pyramid_num_inference_steps_list
    if args.is_amplify_first_chunk:
        extra_args["is_amplify_first_chunk"] = True
    if args.use_cfg_zero_star:
        extra_args["use_cfg_zero_star"] = True
    if args.use_zero_init:
        extra_args["use_zero_init"] = True
        extra_args["zero_steps"] = args.zero_steps

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Helios Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"  Video size: {args.width}x{args.height}, {args.num_frames} frames")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    if args.is_enable_stage2:
        print(f"  Stage 2: enabled (stages={args.pyramid_num_stages}, steps={args.pyramid_num_inference_steps_list})")
        if args.is_amplify_first_chunk:
            print("  DMD amplify first chunk: enabled")
        if args.use_cfg_zero_star:
            print(f"  CFG Zero Star: enabled (zero_init={args.use_zero_init}, zero_steps={args.zero_steps})")
    else:
        if args.use_cfg_zero_star:
            print(f"  CFG Zero Star: enabled (zero_init={args.use_zero_init}, zero_steps={args.zero_steps})")
        print("  Stage 2: disabled (Stage 1 only)")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()
    frames = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            extra_args=extra_args,
        ),
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    # Extract video frames from OmniRequestOutput
    if isinstance(frames, list) and len(frames) > 0:
        first_item = frames[0]

        if hasattr(first_item, "final_output_type"):
            if first_item.final_output_type != "image":
                raise ValueError(
                    f"Unexpected output type '{first_item.final_output_type}', expected 'image' for video generation."
                )

            if hasattr(first_item, "is_pipeline_output") and first_item.is_pipeline_output:
                if isinstance(first_item.request_output, list) and len(first_item.request_output) > 0:
                    inner_output = first_item.request_output[0]
                    if isinstance(inner_output, OmniRequestOutput) and hasattr(inner_output, "images"):
                        frames = inner_output.images[0] if inner_output.images else None
                        if frames is None:
                            raise ValueError("No video frames found in output.")
            elif hasattr(first_item, "images") and first_item.images:
                frames = first_item.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video.")

    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    else:
        video_array = frames
        if hasattr(video_array, "shape") and video_array.ndim == 5:
            video_array = video_array[0]

    if isinstance(video_array, np.ndarray) and video_array.ndim == 4:
        video_array = list(video_array)

    export_to_video(video_array, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
