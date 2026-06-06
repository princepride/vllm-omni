# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC
from typing import ClassVar


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
