# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar


class DiffusionPipelineBase(ABC):
    """Base class for vLLM-Omni diffusion pipelines.

    Pipelines can declare model-specific request and response parameters with
    these class variables. Empty frozensets are valid for models that expose no
    custom parameters.
    """

    # Keys accepted from request-level extra params and copied into
    # OmniDiffusionSamplingParams.extra_args.
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset()

    # Keys copied from DiffusionOutput.custom_output into response metrics.
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset()

    # Some multi-stage pipelines need an ``extra_args`` dict on non-diffusion
    # stages so stage processors can share request-scoped metadata.
    INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES: ClassVar[bool] = False

    @classmethod
    def build_text_to_image_prompt(
        cls,
        prompt: str,
        negative_prompt: str | None,
        height: int | None = None,
        width: int | None = None,
    ) -> dict[str, Any]:
        """Build the shared offline text-to-image prompt for this pipeline."""
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
