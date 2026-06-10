# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_text_to_image_prompt,
    get_extra_body_params,
    get_extra_output_params,
    should_init_extra_args_for_non_diffusion_stages,
)


@pytest.mark.diffusion
@pytest.mark.cpu
def test_bagel_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("BagelPipeline") == frozenset(
        {
            "cfg_text_scale",
            "cfg_img_scale",
            "cfg_interval",
            "cfg_renorm_type",
            "cfg_renorm_min",
            "negative_prompt",
            "think",
            "max_think_tokens",
            "do_sample",
            "text_temperature",
            "timestep_shift",
        }
    )
    assert get_extra_output_params("BagelPipeline") == frozenset({"text_output", "think_text"})
    assert should_init_extra_args_for_non_diffusion_stages("BagelPipeline") is True


@pytest.mark.diffusion
@pytest.mark.cpu
def test_sensenova_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("SenseNovaU1Pipeline") == frozenset(
        {
            "think",
            "cfg_scale",
            "cfg_norm",
            "timestep_shift",
            "t_eps",
            "img_cfg_scale",
            "max_tokens",
        }
    )
    assert get_extra_output_params("SenseNovaU1Pipeline") == frozenset({"think_text"})
    assert should_init_extra_args_for_non_diffusion_stages("SenseNovaU1Pipeline") is False


@pytest.mark.diffusion
@pytest.mark.cpu
def test_unknown_pipeline_has_empty_extra_registry() -> None:
    assert get_extra_body_params("UnknownPipeline") == frozenset()
    assert get_extra_output_params("UnknownPipeline") == frozenset()
    assert should_init_extra_args_for_non_diffusion_stages("UnknownPipeline") is False


@pytest.mark.diffusion
@pytest.mark.cpu
def test_bagel_prompt_builder_is_used() -> None:
    assert build_text_to_image_prompt(
        "BagelPipeline",
        prompt="a cat",
        negative_prompt="blurry",
        height=512,
        width=768,
    ) == {
        "prompt": "<|im_start|>a cat<|im_end|>",
        "modalities": ["image"],
        "mm_processor_kwargs": {
            "target_h": 512,
            "target_w": 768,
            "modalities": ["image"],
        },
        "negative_prompt": "blurry",
    }


@pytest.mark.diffusion
@pytest.mark.cpu
def test_unknown_pipeline_uses_default_prompt_builder() -> None:
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

    declared_extra_params: frozenset[str] = frozenset({"cfg_text_scale", "think"})
    apply_declared_extra_args(
        params,
        declared_extra_params,
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
