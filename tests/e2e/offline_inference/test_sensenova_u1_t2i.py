#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Side-by-side comparison test for SenseNova-U1 T2I generation.

Runs the **original** (HuggingFace / sensenova_u1 package) implementation and
the **vLLM-Omni ported** implementation with identical parameters, then saves
both images and prints pixel-level statistics.

Usage (single-GPU, no TP):
    python tests/e2e/offline_inference/test_sensenova_u1_t2i.py

The script saves:
    outputs/sensenova_u1_ref.png   – reference from original package
    outputs/sensenova_u1_omni.png  – output from vllm-omni port
    outputs/sensenova_u1_diff.png  – amplified difference map
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Shared test config
# ---------------------------------------------------------------------------

PROMPT = (
    "Close portrait of an elderly woman by a farmhouse window, textured skin, "
    "gentle smile, warm natural light, emotional documentary look. The portrait "
    "should feel polished and natural, with sharp eyes, realistic skin texture, "
    "accurate facial anatomy, and premium lighting that keeps the face as the "
    "main focus."
)
WIDTH = 1536
HEIGHT = 2720
CFG_SCALE = 4.0
CFG_NORM = "none"
TIMESTEP_SHIFT = 3.0
NUM_STEPS = 50
SEED = 42
THINK_MODE = False
MODEL_ID = "SenseNova/SenseNova-U1-8B-MoT"
DTYPE = torch.bfloat16
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# 1. Reference: original sensenova_u1 package
# ---------------------------------------------------------------------------


def run_reference(model_path: str) -> Image.Image:
    """Generate an image using the original HuggingFace implementation."""
    print("=" * 60)
    print("[REF] Loading original SenseNova-U1 model …")
    print("=" * 60)

    import sensenova_u1  # noqa: F401 – registers AutoConfig/AutoModel
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, config=config, torch_dtype=DTYPE).to(DEVICE).eval()

    print(f"[REF] Generating {WIDTH}x{HEIGHT}, steps={NUM_STEPS}, seed={SEED} …")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.t2i_generate(
            tokenizer,
            PROMPT,
            image_size=(WIDTH, HEIGHT),
            cfg_scale=CFG_SCALE,
            cfg_norm=CFG_NORM,
            timestep_shift=TIMESTEP_SHIFT,
            cfg_interval=(0.0, 1.0),
            num_steps=NUM_STEPS,
            batch_size=1,
            seed=SEED,
            think_mode=THINK_MODE,
        )
    dt = time.perf_counter() - t0
    print(f"[REF] Done in {dt:.1f}s")

    tensor = out[0] if THINK_MODE else out  # (B, 3, H, W) normalised
    mean = torch.tensor((0.5, 0.5, 0.5), device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = mean.clone()
    arr = ((tensor * std + mean).clamp(0, 1).float().permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(arr[0])

    # Free GPU memory
    del model, tokenizer, config
    torch.cuda.empty_cache()
    return img


# ---------------------------------------------------------------------------
# 2. vLLM-Omni ported version
# ---------------------------------------------------------------------------

_vllm_config_ctx = None


def _init_distributed():
    """Initialize vLLM distributed state for single-GPU standalone usage."""
    import torch.distributed as dist
    from vllm.config.vllm import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    # Set up a minimal VllmConfig so that parallel layers can be constructed.
    global _vllm_config_ctx
    if _vllm_config_ctx is None:
        vllm_cfg = VllmConfig()
        _vllm_config_ctx = set_current_vllm_config(vllm_cfg)
        _vllm_config_ctx.__enter__()

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


def run_omni(model_path: str) -> Image.Image:
    """Generate an image using the vLLM-Omni ported pipeline."""
    print("=" * 60)
    print("[OMNI] Loading vLLM-Omni SenseNova-U1 pipeline …")
    print("=" * 60)

    _init_distributed()

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        SenseNovaU1Pipeline,
    )

    # Build a minimal OmniDiffusionConfig
    od_config = OmniDiffusionConfig()
    od_config.model = model_path
    od_config.dtype = DTYPE
    od_config.revision = None

    pipeline = SenseNovaU1Pipeline(od_config=od_config)

    # Load weights
    print("[OMNI] Loading weights …")
    import glob
    from pathlib import Path

    from safetensors import safe_open

    model_dir = Path(pipeline.local_model_path)

    safetensor_files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not safetensor_files:
        safetensor_files = sorted(glob.glob(str(model_dir / "**/*.safetensors"), recursive=True))

    def _iter_weights():
        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

    loaded = pipeline.load_weights(_iter_weights())
    print(f"[OMNI] Loaded {len(loaded)} parameter tensors")

    pipeline = pipeline.to(device=DEVICE, dtype=DTYPE)
    pipeline.eval()

    # Build a minimal request-like object
    class _FakeSamplingParams:
        height = HEIGHT
        width = WIDTH
        num_inference_steps = NUM_STEPS
        seed = SEED
        extra_args = {
            "cfg_scale": CFG_SCALE,
            "cfg_norm": CFG_NORM,
            "timestep_shift": TIMESTEP_SHIFT,
            "cfg_interval": (0.0, 1.0),
            "batch_size": 1,
            "think": THINK_MODE,
            "t_eps": 0.02,
        }

    class _FakeRequest:
        prompts = [PROMPT]
        sampling_params = _FakeSamplingParams()

    print(f"[OMNI] Generating {WIDTH}x{HEIGHT}, steps={NUM_STEPS}, seed={SEED} …")
    t0 = time.perf_counter()
    result = pipeline.forward(_FakeRequest())
    dt = time.perf_counter() - t0
    print(f"[OMNI] Done in {dt:.1f}s")

    img = result.output
    del pipeline
    torch.cuda.empty_cache()
    return img


# ---------------------------------------------------------------------------
# 3. Comparison
# ---------------------------------------------------------------------------


def compare_images(ref: Image.Image, omni: Image.Image, out_dir: str):
    """Pixel-level comparison and diff visualisation."""
    ref_np = np.array(ref).astype(np.float32)
    omni_np = np.array(omni).astype(np.float32)

    if ref_np.shape != omni_np.shape:
        print(f"[WARN] Shape mismatch: ref={ref_np.shape} vs omni={omni_np.shape}")
        return

    diff = np.abs(ref_np - omni_np)
    mae = diff.mean()
    max_diff = diff.max()
    psnr = 20 * np.log10(255.0 / (np.sqrt(np.mean((ref_np - omni_np) ** 2)) + 1e-8))

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"  Image size     : {ref_np.shape[1]}x{ref_np.shape[0]}")
    print(f"  MAE (0-255)    : {mae:.2f}")
    print(f"  Max diff       : {max_diff:.0f}")
    print(f"  PSNR           : {psnr:.1f} dB")
    print(f"  Exact match    : {np.array_equal(ref_np.astype(np.uint8), omni_np.astype(np.uint8))}")

    # Sample pixel comparisons at specific positions
    positions = [(100, 100), (768, 1360), (400, 600), (1200, 2000), (750, 500)]
    print("\n  Pixel samples (x, y) -> ref_rgb vs omni_rgb:")
    for x, y in positions:
        if y < ref_np.shape[0] and x < ref_np.shape[1]:
            r = tuple(ref_np[y, x].astype(int))
            o = tuple(omni_np[y, x].astype(int))
            d = tuple(diff[y, x].astype(int))
            print(f"    ({x:4d}, {y:4d}): ref={r} omni={o} diff={d}")

    # Diff visualisation (amplified by 5x for visibility)
    diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(os.path.join(out_dir, "sensenova_u1_diff.png"))
    print(f"\n  Diff map saved to {out_dir}/sensenova_u1_diff.png")

    if mae < 1.0:
        print("\n  VERDICT: PASS (MAE < 1.0 — images are essentially identical)")
    elif mae < 10.0:
        print(f"\n  VERDICT: CLOSE (MAE = {mae:.1f} — minor numerical differences)")
    else:
        print(f"\n  VERDICT: DIVERGED (MAE = {mae:.1f} — significant differences, investigate)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="SenseNova-U1 vLLM-Omni comparison test")
    p.add_argument("--model_path", default=MODEL_ID, help="HF model id or local path")
    p.add_argument("--output_dir", default="outputs", help="Where to save images")
    p.add_argument(
        "--mode",
        choices=["both", "ref", "omni"],
        default="both",
        help="'both'=run both and compare, 'ref'=only reference, 'omni'=only vllm-omni",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ref_path = os.path.join(args.output_dir, "sensenova_u1_ref.png")
    omni_path = os.path.join(args.output_dir, "sensenova_u1_omni.png")

    ref_img = None
    omni_img = None

    if args.mode in ("both", "ref"):
        ref_img = run_reference(args.model_path)
        ref_img.save(ref_path)
        print(f"[REF] Saved to {ref_path}")

    if args.mode in ("both", "omni"):
        omni_img = run_omni(args.model_path)
        omni_img.save(omni_path)
        print(f"[OMNI] Saved to {omni_path}")

    if args.mode == "both" and ref_img is not None and omni_img is not None:
        compare_images(ref_img, omni_img, args.output_dir)
    elif args.mode == "omni" and os.path.exists(ref_path):
        print(f"\n[INFO] Found existing reference at {ref_path}, comparing …")
        ref_img = Image.open(ref_path)
        compare_images(ref_img, omni_img, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
