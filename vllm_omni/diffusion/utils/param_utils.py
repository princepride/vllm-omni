# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any


def apply_declared_extra_args(
    sampling_params: Any,
    declared_params: frozenset[str],
    user_kwargs: dict[str, Any],
) -> None:
    """Apply pipeline-declared request params to ``sampling_params.extra_args``.

    This keeps online and offline callers on the same routing contract: standard
    sampling fields stay on ``OmniDiffusionSamplingParams`` while model-specific
    fields declared by the pipeline flow into ``extra_args``.
    """
    if not hasattr(sampling_params, "extra_args"):
        return
    extra_args = getattr(sampling_params, "extra_args", None)
    if extra_args is None:
        extra_args = {}
        setattr(sampling_params, "extra_args", extra_args)
    extra_args.update({key: user_kwargs[key] for key in declared_params if user_kwargs.get(key) is not None})
