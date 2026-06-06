# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import fields
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.models.base import DiffusionPipelineBase
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)

_STANDARD_SAMPLING_PARAM_KEYS = {field.name for field in fields(OmniDiffusionSamplingParams)}


def build_sampling_params(
    pipeline_cls: type[DiffusionPipelineBase],
    height: int | None = None,
    width: int | None = None,
    *,
    seed: int | None = None,
    num_inference_steps: int | None = None,
    **user_kwargs: Any,
) -> OmniDiffusionSamplingParams:
    """Build diffusion sampling params from a pipeline parameter declaration.

    Keys declared in ``pipeline_cls.EXTRA_BODY_PARAMS`` are routed into
    ``extra_args``. Standard ``OmniDiffusionSamplingParams`` fields are applied
    directly, and unknown keys are ignored with a warning.
    """
    sampling_kwargs: dict[str, Any] = {}
    for key, value in (
        ("height", height),
        ("width", width),
        ("seed", seed),
        ("num_inference_steps", num_inference_steps),
    ):
        if value is not None:
            sampling_kwargs[key] = value

    extra_args: dict[str, Any] = {}
    unknown: set[str] = set()
    for key, value in user_kwargs.items():
        if key in pipeline_cls.EXTRA_BODY_PARAMS:
            extra_args[key] = value
        elif key in _STANDARD_SAMPLING_PARAM_KEYS:
            sampling_kwargs[key] = value
        else:
            unknown.add(key)

    if unknown:
        logger.warning("Unknown diffusion sampling params ignored: %s", sorted(unknown))

    return OmniDiffusionSamplingParams(**sampling_kwargs, extra_args=extra_args)
