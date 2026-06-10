# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_omni.model_extras.bagel import (
    BAGEL_EXTRA_BODY_PARAMS,
    BAGEL_EXTRA_OUTPUT_PARAMS,
    BAGEL_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES,
    build_text_to_image_prompt as build_bagel_text_to_image_prompt,
)
from vllm_omni.model_extras.sensenova_u1 import (
    SENSENOVA_U1_EXTRA_BODY_PARAMS,
    SENSENOVA_U1_EXTRA_OUTPUT_PARAMS,
)

TextToImagePromptBuilder = Callable[[str, str | None, int | None, int | None], dict[str, Any]]


def default_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }


_EXTRA_SPECS: dict[str, dict[str, Any]] = {
    "BagelPipeline": {
        "extra_body_params": BAGEL_EXTRA_BODY_PARAMS,
        "extra_output_params": BAGEL_EXTRA_OUTPUT_PARAMS,
        "init_extra_args_for_non_diffusion_stages": BAGEL_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES,
        "text_to_image_prompt_builder": build_bagel_text_to_image_prompt,
    },
    "SenseNovaU1Pipeline": {
        "extra_body_params": SENSENOVA_U1_EXTRA_BODY_PARAMS,
        "extra_output_params": SENSENOVA_U1_EXTRA_OUTPUT_PARAMS,
    },
}


def _get_spec(model_class_name: str | None) -> dict[str, Any] | None:
    if not model_class_name:
        return None
    return _EXTRA_SPECS.get(model_class_name)


def get_extra_body_params(model_class_name: str | None) -> frozenset[str]:
    spec = _get_spec(model_class_name)
    return spec.get("extra_body_params", frozenset()) if spec is not None else frozenset()


def get_extra_output_params(model_class_name: str | None) -> frozenset[str]:
    spec = _get_spec(model_class_name)
    return spec.get("extra_output_params", frozenset()) if spec is not None else frozenset()


def should_init_extra_args_for_non_diffusion_stages(model_class_name: str | None) -> bool:
    spec = _get_spec(model_class_name)
    return bool(spec and spec.get("init_extra_args_for_non_diffusion_stages", False))


def build_text_to_image_prompt(
    model_class_name: str | None,
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    spec = _get_spec(model_class_name)
    builder: TextToImagePromptBuilder = (
        spec.get("text_to_image_prompt_builder", default_text_to_image_prompt)
        if spec is not None
        else default_text_to_image_prompt
    )
    return builder(prompt, negative_prompt, height, width)
