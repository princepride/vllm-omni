# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Helios (https://github.com/BestWishYsh/Helios)

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel
from vllm_omni.diffusion.models.helios.scheduling_helios import HeliosScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(config: dict) -> HeliosTransformer3DModel:
    kwargs = {}

    key_map = [
        "patch_size",
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "text_dim",
        "freq_dim",
        "ffn_dim",
        "num_layers",
        "cross_attn_norm",
        "qk_norm",
        "eps",
        "added_kv_proj_dim",
        "rope_dim",
        "rope_theta",
        "guidance_cross_attn",
        "zero_history_timestep",
        "has_multi_term_memory_patch",
        "is_amplify_history",
        "history_scale_mode",
    ]
    for key in key_map:
        if key in config:
            val = config[key]
            if key in ("patch_size", "rope_dim") and isinstance(val, list):
                val = tuple(val)
            kwargs[key] = val

    return HeliosTransformer3DModel(**kwargs)


def get_helios_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_helios_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        return request

    return pre_process_func


class HeliosPipeline(nn.Module, CFGParallelMixin):
    """Helios text-to-video pipeline for vllm-omni.

    Implements chunked video generation with multi-term memory history context.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
        ).to(self.device)

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(transformer_config)

        # Read scheduler config to determine scheduler type
        scheduler_config_path = os.path.join(model, "scheduler", "scheduler_config.json") if local_files_only else None
        scheduler_kwargs = {}
        if scheduler_config_path and os.path.exists(scheduler_config_path):
            with open(scheduler_config_path) as f:
                sched_cfg = json.load(f)
                passthrough_keys = [
                    "num_train_timesteps",
                    "shift",
                    "stages",
                    "stage_range",
                    "gamma",
                    "thresholding",
                    "prediction_type",
                    "solver_order",
                    "predict_x0",
                    "solver_type",
                    "lower_order_final",
                    "disable_corrector",
                    "use_flow_sigmas",
                    "scheduler_type",
                    "use_dynamic_shifting",
                    "time_shift_type",
                ]
                for key in passthrough_keys:
                    if key in sched_cfg:
                        scheduler_kwargs[key] = sched_cfg[key]

        self.scheduler = HeliosScheduler(**scheduler_kwargs)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 384,
        width: int = 640,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        frame_num: int = 132,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        # Helios-specific
        history_sizes: list | None = None,
        num_latent_frames_per_chunk: int = 9,
        keep_first_frame: bool = True,
        **kwargs,
    ) -> DiffusionOutput:
        if history_sizes is None:
            history_sizes = [16, 2, 1]

        if len(req.prompts) > 1:
            raise ValueError("This model only supports a single prompt, not a batched request.")
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Helios generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale

        height = (height // 16) * 16
        width = (width // 16) * 16
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Encode prompts
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
                max_sequence_length=req.sampling_params.max_sequence_length or 226,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        batch_size = prompt_embeds.shape[0]

        history_sizes = sorted(history_sizes, reverse=True)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )

        # Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        window_num_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
        num_latent_chunk = max(1, (num_frames + window_num_frames - 1) // window_num_frames)
        history_video = None
        total_generated_latent_frames = 0
        image_latents = None

        if not keep_first_frame:
            history_sizes[-1] = history_sizes[-1] + 1

        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )

        # Prepare frame indices
        if keep_first_frame:
            indices = torch.arange(0, sum([1, *history_sizes, num_latent_frames_per_chunk]))
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
            indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
        else:
            indices = torch.arange(0, sum([*history_sizes, num_latent_frames_per_chunk]))
            (
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_short,
                indices_hidden_states,
            ) = indices.split([*history_sizes, num_latent_frames_per_chunk], dim=0)

        indices_hidden_states = indices_hidden_states.unsqueeze(0)
        indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

        if attention_kwargs is None:
            attention_kwargs = {}

        vae_dtype = self.vae.dtype

        # Chunked denoising loop
        for k in range(num_latent_chunk):
            is_first_chunk = k == 0

            if keep_first_frame:
                latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                    :, :, -sum(history_sizes) :
                ].split(history_sizes, dim=2)
                if image_latents is None and is_first_chunk:
                    latents_prefix = torch.zeros(
                        (
                            batch_size,
                            num_channels_latents,
                            1,
                            latents_history_1x.shape[-2],
                            latents_history_1x.shape[-1],
                        ),
                        device=latents_history_1x.device,
                        dtype=latents_history_1x.dtype,
                    )
                else:
                    latents_prefix = image_latents
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                latents_history_long, latents_history_mid, latents_history_short = history_latents[
                    :, :, -sum(history_sizes) :
                ].split(history_sizes, dim=2)

            # Prepare noise for this chunk
            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                window_num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            # Set up scheduler timesteps
            patch_size = self.transformer.config.patch_size
            image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                patch_size[0] * patch_size[1] * patch_size[2]
            )
            sigmas = np.linspace(0.999, 0.0, num_steps + 1)[:-1]
            mu = calculate_shift(image_seq_len)
            self.scheduler.set_timesteps(num_steps, device=device, sigmas=sigmas, mu=mu)
            timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            # Stage 1 denoising
            latents = self._stage1_sample(
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                attention_kwargs=attention_kwargs,
                transformer_dtype=dtype,
            )

            if keep_first_frame and is_first_chunk and image_latents is None:
                image_latents = latents[:, :, 0:1, :, :]

            total_generated_latent_frames += latents.shape[2]
            history_latents = torch.cat([history_latents, latents], dim=2)
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
            index_slice = (
                slice(None),
                slice(None),
                slice(-num_latent_frames_per_chunk, None),
            )

            current_latents = real_history_latents[index_slice].to(vae_dtype) / latents_std + latents_mean
            current_video = self.vae.decode(current_latents, return_dict=False)[0]

            if history_video is None:
                history_video = current_video
            else:
                history_video = torch.cat([history_video, current_video], dim=2)

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        if output_type == "latent":
            output = real_history_latents
        else:
            output = history_video

        return DiffusionOutput(output=output)

    def _stage1_sample(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        timesteps: torch.Tensor,
        guidance_scale: float,
        indices_hidden_states: torch.Tensor,
        indices_latents_history_short: torch.Tensor,
        indices_latents_history_mid: torch.Tensor,
        indices_latents_history_long: torch.Tensor,
        latents_history_short: torch.Tensor,
        latents_history_mid: torch.Tensor,
        latents_history_long: torch.Tensor,
        attention_kwargs: dict,
        transformer_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Single-stage denoising loop for one chunk."""
        for t in timesteps:
            self._current_timestep = t
            timestep = t.expand(latents.shape[0])

            do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

            positive_kwargs = {
                "hidden_states": latents.to(transformer_dtype),
                "timestep": timestep,
                "encoder_hidden_states": prompt_embeds,
                "indices_hidden_states": indices_hidden_states,
                "indices_latents_history_short": indices_latents_history_short,
                "indices_latents_history_mid": indices_latents_history_mid,
                "indices_latents_history_long": indices_latents_history_long,
                "latents_history_short": latents_history_short.to(transformer_dtype),
                "latents_history_mid": latents_history_mid.to(transformer_dtype),
                "latents_history_long": latents_history_long.to(transformer_dtype),
                "attention_kwargs": attention_kwargs,
                "return_dict": False,
            }

            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latents.to(transformer_dtype),
                    "timestep": timestep,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "indices_hidden_states": indices_hidden_states,
                    "indices_latents_history_short": indices_latents_history_short,
                    "indices_latents_history_mid": indices_latents_history_mid,
                    "indices_latents_history_long": indices_latents_history_long,
                    "latents_history_short": latents_history_short.to(transformer_dtype),
                    "latents_history_mid": latents_history_mid.to(transformer_dtype),
                    "latents_history_long": latents_history_long.to(transformer_dtype),
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )

            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

        return latents

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)[0]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
