import argparse
import os
import sys

from datasets import load_dataset
from tqdm import tqdm

# Ensure current directory is in path for imports
sys.path.append(os.getcwd())

from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    TeaCacheCoefficientEstimator,
)


def main():
    parser = argparse.ArgumentParser(description="Estimate TeaCache coefficients for T2I models.")
    parser.add_argument("--model", type=str, default="../models/BAGEL-7B-MoT", help="Path to the model")
    parser.add_argument("--model_type", type=str, default="Bagel", help="Model type (default: Bagel)")
    parser.add_argument("--num_prompts", type=int, default=70, help="Number of prompts to sample (paper suggests ~70)")
    parser.add_argument("--poly_order", type=int, default=4, help="Order of polynomial fitting (default: 4)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (default: cuda)")
    args = parser.parse_args()

    print(f"Loading model from {args.model} (type: {args.model_type})...")
    estimator = TeaCacheCoefficientEstimator(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
    )

    print("Loading nateraw/parti-prompts dataset...")
    # https://github.com/ali-vilab/TeaCache/issues/20#issuecomment-2574651021
    dataset = load_dataset("nateraw/parti-prompts", split="train")

    # Shuffle or just take the first N
    prompts = dataset["Prompt"][: args.num_prompts]

    print(f"Starting data collection with {len(prompts)} prompts...")
    for i, prompt in enumerate(tqdm(prompts)):
        estimator.collect_from_prompt(
            prompt,
            num_inference_steps=args.num_inference_steps,
        )

    print("Estimating coefficients...")
    try:
        coeffs = estimator.estimate(poly_order=args.poly_order)
        print("\n" + "=" * 40)
        print(f"Estimated Coefficients (Model: {estimator.transformer_type})")
        print("=" * 40)
        print("Format: [an, ..., a1, a0]")
        print("-" * 40)

        coeffs_str = "[\n" + ",\n".join([f"    {c:.8e}" for c in coeffs]) + "\n]"
        print(coeffs_str)
        print("-" * 40)

        print("\nRaw list:")
        print(coeffs)

    except ValueError as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
