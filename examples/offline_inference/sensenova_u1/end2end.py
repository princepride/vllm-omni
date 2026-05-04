# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end offline inference for SenseNova-U1-8B-MoT text-to-image generation.

SenseNova-U1 is a unified Qwen3-based LLM with Mixture-of-Tokenizers (MoT)
attention that handles text encoding, optional think-mode reasoning, and
flow-matching-based image denoising in a single pipeline.

Usage:
    python end2end.py --prompt "A cute cat" --think
    python end2end.py --prompt "A futuristic cityscape" --width 2048 --height 2048

See README.md for more examples.
"""

import argparse
import os

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseNova-U1 text-to-image generation via vLLM-Omni.",
    )
    parser.add_argument(
        "--model",
        default="SenseNova/SenseNova-U1-8B-MoT",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="A cute cat sitting on a windowsill, soft natural light",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory to save generated images.",
    )
    # Image dimensions
    parser.add_argument("--height", type=int, default=2048, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=2048, help="Width of generated image.")

    # Generation parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--cfg-norm",
        type=str,
        default="none",
        help="CFG normalization mode.",
    )
    parser.add_argument(
        "--timestep-shift",
        type=float,
        default=3.0,
        help="Timestep shift for flow-matching schedule.",
    )
    parser.add_argument(
        "--t-eps",
        type=float,
        default=0.02,
        help="Epsilon for flow-matching timestep schedule.",
    )

    # Think mode
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable think mode: the model reasons about the prompt before generating the image.",
    )
    parser.add_argument(
        "--print-think",
        action="store_true",
        help="Print the generated think text to stdout.",
    )

    # Advanced
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    omni = Omni(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_inference_steps=args.num_steps,
        extra_args={
            "cfg_scale": args.cfg_scale,
            "cfg_norm": args.cfg_norm,
            "timestep_shift": args.timestep_shift,
            "cfg_interval": (0.0, 1.0),
            "batch_size": 1,
            "think": args.think,
            "t_eps": args.t_eps,
        },
    )

    print(f"\n{'=' * 60}")
    print("SenseNova-U1 Generation Configuration:")
    print(f"  Model       : {args.model}")
    print(f"  Image size  : {args.width}x{args.height}")
    print(f"  Steps       : {args.num_steps}")
    print(f"  CFG scale   : {args.cfg_scale}")
    print(f"  Seed        : {args.seed}")
    print(f"  Think mode  : {args.think}")
    print(f"  TP size     : {args.tensor_parallel_size}")
    print(f"{'=' * 60}\n")

    outputs = list(
        omni.generate(
            prompts={"prompt": args.prompt, "modalities": ["image"]},
            sampling_params_list=sampling_params,
        )
    )

    for req_output in outputs:
        # Single-stage DiT: think text lives on `_custom_output`, images on the request output directly.
        custom = getattr(req_output, "_custom_output", {}) or {}
        if args.print_think and custom.get("think_text"):
            print(f"[Think]\n{custom['think_text']}\n")

        images = getattr(req_output, "images", None) or []
        if not images:
            print("[Warning] No images generated.")
            continue

        for j, img in enumerate(images):
            save_path = os.path.join(args.output, f"sensenova_u1_output_{j}.png")
            img.save(save_path)
            print(f"[Output] Saved {img.size[0]}x{img.size[1]} image to {save_path}")


if __name__ == "__main__":
    main()
