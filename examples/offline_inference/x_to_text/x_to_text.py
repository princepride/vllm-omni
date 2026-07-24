# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared offline example for any supported input modality to text."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoTokenizer

from vllm_omni import Omni

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HUNYUAN_AR_DEPLOY_CONFIG = _REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_ar.yaml"
_MAMMOTH_AR_DEPLOY_CONFIG = _REPO_ROOT / "vllm_omni" / "deploy" / "mammoth_moda2_ar.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared offline x-to-text inference (currently T2T and I2T).")
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument("--prompt", required=True, help="Question or text prompt.")
    parser.add_argument("--image", help="Optional input image. Supplying it selects image-to-text.")
    parser.add_argument("--output", help="Optional path to write the generated text.")
    parser.add_argument("--deploy-config", help="Optional deploy YAML override.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", default=None)
    parser.add_argument("--log-stats", action="store_true")
    return parser.parse_args()


def _model_family(model: str, trust_remote_code: bool) -> str:
    del trust_remote_code  # Reading config.json does not execute checkpoint code.
    from vllm.transformers_utils.config import get_hf_file_to_dict

    config = get_hf_file_to_dict("config.json", model) or {}
    model_type = str(config.get("model_type", "")).lower()
    architectures = {str(value).lower() for value in (config.get("architectures") or [])}
    if model_type == "bagel" or "bagelforconditionalgeneration" in architectures:
        return "bagel"
    if model_type == "hunyuan_image_3_moe" or any("hunyuanimage3" in value for value in architectures):
        return "hunyuan_image3"
    if "mammoth" in model_type or any("mammothmoda2" in value for value in architectures):
        return "mammoth_moda2"
    return "generic"


def _bagel_prompt(prompt: str, has_image: bool) -> dict[str, Any]:
    image_token = "<|image_pad|>\n" if has_image else ""
    return {
        "prompt": f"<|im_start|>user\n{image_token}{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "modalities": ["text"],
    }


def _mammoth_prompt(prompt: str, has_image: bool) -> dict[str, Any]:
    vision = "<|vision_start|><|image_pad|><|vision_end|>" if has_image else ""
    return {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{vision}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "modalities": ["text"],
        "additional_information": {"omni_task": ["chat"]},
    }


def _hunyuan_prompt(model: str, prompt: str, has_image: bool) -> tuple[dict[str, Any], list[int]]:
    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
        resolve_stop_token_ids,
        resolve_sys_type,
    )

    task = "i2t" if has_image else "t2t"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    build_kwargs: dict[str, Any] = {"task": task, "bot_task": None}
    if has_image:
        build_kwargs["num_images"] = 1
    built = build_prompt_tokens(prompt, tokenizer, **build_kwargs)
    return (
        {
            "prompt": prompt,
            "prompt_token_ids": built.token_ids,
            "modalities": ["text"],
            "use_system_prompt": resolve_sys_type(None),
        },
        resolve_stop_token_ids(task=task, bot_task=None, tokenizer=tokenizer, image_size="auto"),
    )


def _extract_text(outputs: list[Any]) -> str:
    chunks: list[str] = []
    for output in outputs:
        request_output = getattr(output, "request_output", output)
        for completion in getattr(request_output, "outputs", None) or []:
            chunks.append(getattr(completion, "text", "") or "")
    return "".join(chunks).strip()


def main() -> None:
    args = parse_args()
    family = _model_family(args.model, args.trust_remote_code)
    image = Image.open(args.image).convert("RGB") if args.image else None

    if family == "bagel":
        prompt_dict = _bagel_prompt(args.prompt, image is not None)
        stop_token_ids = None
    elif family == "hunyuan_image3":
        prompt_dict, stop_token_ids = _hunyuan_prompt(args.model, args.prompt, image is not None)
    elif family == "mammoth_moda2":
        prompt_dict = _mammoth_prompt(args.prompt, image is not None)
        stop_token_ids = None
    else:
        prompt_dict = {"prompt": args.prompt, "modalities": ["text"]}
        stop_token_ids = None

    if image is not None:
        prompt_dict["multi_modal_data"] = {"image": image}

    omni_kwargs: dict[str, Any] = {
        "model": args.model,
        "mode": "image-to-text" if image is not None else "text-to-text",
        "trust_remote_code": args.trust_remote_code or family in {"hunyuan_image3", "mammoth_moda2"},
        "log_stats": args.log_stats,
    }
    if args.enforce_eager is not None:
        omni_kwargs["enforce_eager"] = args.enforce_eager
    if args.deploy_config:
        omni_kwargs["deploy_config"] = args.deploy_config
    elif family == "hunyuan_image3":
        omni_kwargs["deploy_config"] = str(_HUNYUAN_AR_DEPLOY_CONFIG)
    elif family == "mammoth_moda2":
        omni_kwargs["deploy_config"] = str(_MAMMOTH_AR_DEPLOY_CONFIG)

    omni = Omni(**omni_kwargs)
    try:
        sampling_params = list(omni.default_sampling_params_list or [])
        for params in sampling_params:
            if hasattr(params, "max_tokens"):
                params.max_tokens = args.max_tokens
            if hasattr(params, "temperature"):
                params.temperature = args.temperature
            if hasattr(params, "top_p"):
                params.top_p = args.top_p
            if hasattr(params, "seed"):
                params.seed = args.seed
            if stop_token_ids is not None and hasattr(params, "stop_token_ids"):
                params.stop_token_ids = stop_token_ids
        outputs = list(omni.generate([prompt_dict], sampling_params_list=sampling_params or None))
    finally:
        omni.close()

    text = _extract_text(outputs)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
