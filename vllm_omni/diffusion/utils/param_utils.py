# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def apply_declared_extra_args(
    sampling_params: OmniDiffusionSamplingParams,
    declared_params: frozenset[str],
    user_kwargs: dict[str, object],
) -> None:
    """Route pipeline-declared request params into ``sampling_params.extra_args``.

    Both online serving and offline examples call this so that model-specific
    keys (e.g. ``cfg_text_scale`` for BAGEL) end up in ``extra_args`` instead
    of being silently dropped.
    """
    sampling_params.extra_args.update(
        {key: user_kwargs[key] for key in declared_params if user_kwargs.get(key) is not None}
    )
