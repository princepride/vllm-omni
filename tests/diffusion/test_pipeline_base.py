# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import ClassVar

import pytest

from vllm_omni.diffusion.diffusion_engine import (
    get_extra_body_params,
    get_extra_output_params,
)
from vllm_omni.diffusion.models.base import DiffusionPipelineBase
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class _ValidPipeline(DiffusionPipelineBase):
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset({"cfg_text_scale", "think"})
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset({"think_text"})


class _EmptyPipeline(DiffusionPipelineBase):
    pass


class _LegacyPipeline:
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset({"legacy_param"})
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset({"legacy_output"})


@pytest.fixture
def patch_registry(monkeypatch: pytest.MonkeyPatch):
    models = {
        "ValidPipeline": _ValidPipeline,
        "EmptyPipeline": _EmptyPipeline,
        "LegacyPipeline": _LegacyPipeline,
    }
    monkeypatch.setattr(
        DiffusionModelRegistry,
        "_try_load_model_cls",
        staticmethod(lambda model_class_name: models.get(model_class_name)),
    )
    return models


@pytest.mark.diffusion
@pytest.mark.cpu
def test_valid_pipeline_uses_declared_params(patch_registry) -> None:
    assert get_extra_body_params("ValidPipeline") == frozenset({"cfg_text_scale", "think"})
    assert get_extra_output_params("ValidPipeline") == frozenset({"think_text"})


@pytest.mark.diffusion
@pytest.mark.cpu
def test_empty_frozensets_are_valid(patch_registry) -> None:
    assert get_extra_body_params("EmptyPipeline") == frozenset()
    assert get_extra_output_params("EmptyPipeline") == frozenset()


@pytest.mark.diffusion
@pytest.mark.cpu
def test_legacy_pipeline_falls_back_during_migration(patch_registry) -> None:
    assert get_extra_body_params("LegacyPipeline") == frozenset({"legacy_param"})
    assert get_extra_output_params("LegacyPipeline") == frozenset({"legacy_output"})


@pytest.mark.diffusion
@pytest.mark.cpu
def test_declared_extra_args_apply_to_existing_sampling_params() -> None:
    params = OmniDiffusionSamplingParams(extra_args={"existing": 1})

    apply_declared_extra_args(
        params,
        _ValidPipeline.EXTRA_BODY_PARAMS,
        {
            "cfg_text_scale": 4.0,
            "think": False,
            "unknown": "ignored",
        },
    )

    assert params.extra_args == {
        "existing": 1,
        "cfg_text_scale": 4.0,
        "think": False,
    }
