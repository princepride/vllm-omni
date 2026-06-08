# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from vllm_omni.diffusion.diffusion_engine import (
    build_text_to_image_prompt,
    get_extra_body_params,
    get_extra_output_params,
    should_init_extra_args_for_non_diffusion_stages,
)
from vllm_omni.diffusion.models.base import DiffusionPipelineBase
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class _ValidPipeline(DiffusionPipelineBase):
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset({"cfg_text_scale", "think"})
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset({"think_text"})


class _PromptPipeline(DiffusionPipelineBase):
    INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES: ClassVar[bool] = True

    @classmethod
    def build_text_to_image_prompt(
        cls,
        prompt: str,
        negative_prompt: str | None,
        height: int | None = None,
        width: int | None = None,
    ) -> dict[str, Any]:
        return {
            "prompt": f"<wrapped>{prompt}</wrapped>",
            "negative_prompt": negative_prompt,
            "modalities": ["image"],
            "mm_processor_kwargs": {
                "target_h": height,
                "target_w": width,
            },
        }


class _EmptyPipeline(DiffusionPipelineBase):
    pass


class _LegacyPipeline:
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset({"legacy_param"})
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset({"legacy_output"})


@pytest.fixture
def patch_registry(monkeypatch: pytest.MonkeyPatch):
    models = {
        "ValidPipeline": _ValidPipeline,
        "PromptPipeline": _PromptPipeline,
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
    assert should_init_extra_args_for_non_diffusion_stages("EmptyPipeline") is False


@pytest.mark.diffusion
@pytest.mark.cpu
def test_legacy_pipeline_falls_back_during_migration(patch_registry) -> None:
    assert get_extra_body_params("LegacyPipeline") == frozenset({"legacy_param"})
    assert get_extra_output_params("LegacyPipeline") == frozenset({"legacy_output"})


@pytest.mark.diffusion
@pytest.mark.cpu
def test_pipeline_declared_prompt_builder_is_used(patch_registry) -> None:
    assert build_text_to_image_prompt(
        "PromptPipeline",
        prompt="a cat",
        negative_prompt="blurry",
        height=512,
        width=768,
    ) == {
        "prompt": "<wrapped>a cat</wrapped>",
        "negative_prompt": "blurry",
        "modalities": ["image"],
        "mm_processor_kwargs": {
            "target_h": 512,
            "target_w": 768,
        },
    }
    assert should_init_extra_args_for_non_diffusion_stages("PromptPipeline") is True


@pytest.mark.diffusion
@pytest.mark.cpu
def test_unknown_pipeline_uses_default_prompt_builder(patch_registry) -> None:
    assert build_text_to_image_prompt(
        "UnknownPipeline",
        prompt="a cat",
        negative_prompt=None,
        height=512,
        width=512,
    ) == {
        "prompt": "a cat",
        "negative_prompt": None,
    }


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
