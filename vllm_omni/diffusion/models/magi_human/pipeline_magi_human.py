# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from daVinci-MagiHuman pipeline.
# Copyright (c) 2026 SandAI. All Rights Reserved.

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import numpy as np
import PIL.Image
import torch
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .magi_human_data_proxy import MagiDataProxy
from .magi_human_dit import DiTModel, MagiHumanDiTConfig
from .magi_human_scheduler import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)

# Default negative prompt from original MagiHuman
_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
    ", low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, "
    "static, compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, "
    "unclear, echo, reverb, room echo, over-reverberated, hollow sound, distant, washed out, harsh, "
    "shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, "
    "abrupt cut, jarring transition, sudden silence, looping artifact, music, instrumental, sirens, "
    "alarms, crowd noise, unrelated sound effects, chaotic, disorganized, messy, cheap sound"
    ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, "
    "flat intonation, undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, "
    "TTS, insincere, fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, "
    "hesitant, unconfident, tired, weak voice, stuttering, stammering, mumbling, slurred speech, "
    "mispronounced, bad articulation, lisp, vocal fry, creaky voice, mouth clicks, lip smacks, "
    "wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, coughing, clearing throat, "
    "sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural pauses, "
    "awkward silence, choppy, disjointed, multiple speakers, two voices, background talking, "
    "out of tune, off-key, autotune artifacts"
)


@dataclass
class EvalInput:
    """Input data for a single DiT forward pass."""

    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | list[int]


class ZeroSNRDDPMDiscretization:
    """Discretization for zero-SNR DDPM noise schedule."""

    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,
        keep_start=False,
        post_shift=False,
    ):
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        self.num_timesteps = num_timesteps
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64) ** 2
        betas = betas.cpu().numpy()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)
        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)
        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def __call__(self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False):
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])]) if do_append_zero else sigmas
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))

    def get_sigmas(self, n, device="cpu", return_idx=False):
        if n < self.num_timesteps:
            timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            return torch.flip(alphas_cumprod_sqrt, (0,))


def pad_or_trim(tensor: torch.Tensor, target_size: int, dim: int, pad_value: float = 0.0):
    current_size = tensor.size(dim)
    if current_size < target_size:
        padding_amount = target_size - current_size
        padding_tuple = [0] * (2 * tensor.dim())
        padding_dim_index = tensor.dim() - 1 - dim
        padding_tuple[2 * padding_dim_index + 1] = padding_amount
        return torch.nn.functional.pad(tensor, tuple(padding_tuple), "constant", pad_value), current_size
    slicing = [slice(None)] * tensor.dim()
    slicing[dim] = slice(0, target_size)
    return tensor[tuple(slicing)], target_size


def schedule_latent_step(
    *,
    video_scheduler,
    audio_scheduler,
    latent_video,
    latent_audio,
    t,
    idx,
    steps,
    v_cfg_video,
    v_cfg_audio,
    is_a2v,
    cfg_number,
    use_sr_model,
    using_sde_flag,
):
    if cfg_number == 1 and not use_sr_model:
        latent_video = video_scheduler.step_ddim(v_cfg_video, idx, latent_video)
        latent_audio = audio_scheduler.step_ddim(v_cfg_audio, idx, latent_audio)
        return latent_video, latent_audio

    if using_sde_flag:
        if use_sr_model:
            latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
            return latent_video, latent_audio
        if idx < int(steps * (3 / 4)):
            noise_theta = 1.0 if (idx + 1) % 2 == 0 else 0.0
        else:
            noise_theta = 1.0 if idx % 3 == 0 else 0.0
        latent_video = video_scheduler.step_sde(v_cfg_video, idx, latent_video, noise_theta=noise_theta)
        if not is_a2v:
            latent_audio = audio_scheduler.step_sde(v_cfg_audio, idx, latent_audio, noise_theta=noise_theta)
        return latent_video, latent_audio

    latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
    if not is_a2v and not use_sr_model:
        latent_audio = audio_scheduler.step(v_cfg_audio, t, latent_audio, return_dict=False)[0]
    return latent_video, latent_audio


def _resizecrop(img, height, width):
    """Resize and center-crop to target dimensions."""
    img_w, img_h = img.size
    target_ratio = width / height
    img_ratio = img_w / img_h
    if img_ratio > target_ratio:
        new_h = img_h
        new_w = int(img_h * target_ratio)
    else:
        new_w = img_w
        new_h = int(img_w / target_ratio)
    left = (img_w - new_w) // 2
    top = (img_h - new_h) // 2
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Pre/post process functions for registry
# ---------------------------------------------------------------------------
def get_magi_human_post_process_func(od_config: OmniDiffusionConfig):
    """Post-process: video tensor + audio numpy output."""
    video_processor = VideoProcessor(vae_scale_factor=16)

    def post_process_func(output: dict, output_type: str = "np"):
        video = output.get("video")
        audio = output.get("audio")
        if video is not None and output_type != "latent" and isinstance(video, torch.Tensor):
            video = video_processor.postprocess_video(video, output_type=output_type)
        return {"video": video, "audio": audio}

    return post_process_func


def get_magi_human_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process: load and resize input image/audio."""

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                continue
            multi_modal_data = prompt.get("multi_modal_data", {})
            raw_image = multi_modal_data.get("image", None)
            if raw_image is not None and isinstance(raw_image, str):
                image = PIL.Image.open(raw_image).convert("RGB")
                multi_modal_data["image"] = image
                prompt["multi_modal_data"] = multi_modal_data
                request.prompts[i] = prompt
        return request

    return pre_process_func


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
class MagiHumanPipeline(nn.Module, ProgressBarMixin, DiffusionPipelineProfilerMixin):
    """MagiHuman video+audio generation pipeline for vLLM-Omni.

    This pipeline generates video and audio jointly from text/image/audio inputs
    using a DiT (Diffusion Transformer) architecture with multimodal token routing.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        model_path = od_config.model
        local_files_only = os.path.exists(model_path)

        # ---- Load pipeline config ----
        pipeline_config = {}
        config_path = os.path.join(model_path, "pipeline_config.json") if local_files_only else None
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                pipeline_config = json.load(f)

        # ---- DiT config ----
        dit_config_path = os.path.join(model_path, "dit", "config.json") if local_files_only else None
        dit_config_dict = {}
        if dit_config_path and os.path.exists(dit_config_path):
            with open(dit_config_path) as f:
                dit_config_dict = json.load(f)

        dit_config = (
            MagiHumanDiTConfig(
                **{k: v for k, v in dit_config_dict.items() if k in MagiHumanDiTConfig.__dataclass_fields__}
            )
            if dit_config_dict
            else MagiHumanDiTConfig()
        )

        # ---- Build DiT model ----
        self.dit = DiTModel(model_config=dit_config)

        # ---- SR DiT model (optional) ----
        self.sr_dit = None
        sr_dit_path = os.path.join(model_path, "sr_dit") if local_files_only else None
        if sr_dit_path and os.path.exists(sr_dit_path):
            sr_config_path = os.path.join(sr_dit_path, "config.json")
            sr_config_dict = {}
            if os.path.exists(sr_config_path):
                with open(sr_config_path) as f:
                    sr_config_dict = json.load(f)
            sr_dit_config = (
                MagiHumanDiTConfig(
                    **{k: v for k, v in sr_config_dict.items() if k in MagiHumanDiTConfig.__dataclass_fields__}
                )
                if sr_config_dict
                else MagiHumanDiTConfig()
            )
            # SR model typically has local attention layers
            if not sr_dit_config.local_attn_layers:
                sr_dit_config.local_attn_layers = [
                    0,
                    1,
                    2,
                    4,
                    5,
                    6,
                    8,
                    9,
                    10,
                    12,
                    13,
                    14,
                    16,
                    17,
                    18,
                    20,
                    21,
                    22,
                    24,
                    25,
                    26,
                    28,
                    29,
                    30,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                ]
            self.sr_dit = DiTModel(model_config=sr_dit_config)

        # ---- Text encoder (T5-Gemma) ----
        # Loaded via from_pretrained from HuggingFace format
        text_encoder_path = os.path.join(model_path, "text_encoder") if local_files_only else None
        self.text_encoder = None
        self.tokenizer = None
        if text_encoder_path and os.path.exists(text_encoder_path):
            try:
                from transformers.models.t5gemma import T5GemmaEncoderModel

                self.text_encoder = T5GemmaEncoderModel.from_pretrained(
                    text_encoder_path, is_encoder_decoder=False, dtype=torch.bfloat16
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            except ImportError:
                logger.warning("T5GemmaEncoderModel not available, text encoder will not be loaded.")

        # ---- VAE (Wan2.2 VAE) ----
        self.vae = None
        vae_path = os.path.join(model_path, "vae") if local_files_only else None
        if vae_path and os.path.exists(vae_path):
            try:
                from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
                    DistributedAutoencoderKLWan,
                )

                self.vae = DistributedAutoencoderKLWan.from_pretrained(
                    model_path, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
                ).to(self.device)
            except Exception:
                logger.warning("Failed to load Wan2.2 VAE, trying standalone VAE loading.")

        # ---- Audio VAE (Stable Audio) ----
        self.audio_vae = None
        audio_vae_path = os.path.join(model_path, "audio_vae") if local_files_only else None
        if audio_vae_path and os.path.exists(audio_vae_path):
            try:
                from .magi_human_audio_vae import SAAudioFeatureExtractor

                self.audio_vae = SAAudioFeatureExtractor(device=str(self.device), model_path=audio_vae_path)
            except ImportError:
                logger.warning("Audio VAE module not available.")

        # ---- Weights sources for HuggingFace format loading ----
        self.weights_sources = []
        self.weights_sources.append(
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="dit",
                revision=None,
                prefix="dit.",
                fall_back_to_pt=True,
            )
        )
        if self.sr_dit is not None:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=model_path,
                    subfolder="sr_dit",
                    revision=None,
                    prefix="sr_dit.",
                    fall_back_to_pt=True,
                )
            )

        # ---- Evaluation config defaults ----
        eval_config = pipeline_config.get("evaluation", {})
        self.fps = eval_config.get("fps", 25)
        self.vae_stride = tuple(eval_config.get("vae_stride", (4, 16, 16)))
        self.z_dim = eval_config.get("z_dim", 48)
        self.patch_size = tuple(eval_config.get("patch_size", (1, 2, 2)))
        self.video_txt_guidance_scale = eval_config.get("video_txt_guidance_scale", 5.0)
        self.audio_txt_guidance_scale = eval_config.get("audio_txt_guidance_scale", 5.0)
        self.sr_video_txt_guidance_scale = eval_config.get("sr_video_txt_guidance_scale", 3.5)
        self.shift = eval_config.get("shift", 5.0)
        self.cfg_number = eval_config.get("cfg_number", 2)
        self.sr_cfg_number = eval_config.get("sr_cfg_number", 2)
        self.noise_value = eval_config.get("noise_value", 220)
        self.use_cfg_trick = eval_config.get("use_cfg_trick", True)
        self.cfg_trick_start_frame = eval_config.get("cfg_trick_start_frame", 13)
        self.cfg_trick_value = eval_config.get("cfg_trick_value", 2.0)
        self.using_sde_flag = eval_config.get("using_sde_flag", False)
        self.sr_audio_noise_scale = eval_config.get("sr_audio_noise_scale", 0.7)
        self.t5_gemma_target_length = eval_config.get("t5_gemma_target_length", 640)

        # ---- Data proxy ----
        dp_config = pipeline_config.get("data_proxy", {})
        self.data_proxy = MagiDataProxy(
            patch_size=dp_config.get("patch_size", 2),
            t_patch_size=dp_config.get("t_patch_size", 1),
            frame_receptive_field=dp_config.get("frame_receptive_field", 11),
            spatial_rope_interpolation=dp_config.get("spatial_rope_interpolation", "extra"),
            ref_audio_offset=dp_config.get("ref_audio_offset", 1000),
            text_offset=dp_config.get("text_offset", 0),
            coords_style=dp_config.get("coords_style", "v2"),
        )
        sr_dp_config = copy.deepcopy(dp_config)
        sr_dp_config["coords_style"] = "v1"
        self.sr_data_proxy = MagiDataProxy(
            patch_size=sr_dp_config.get("patch_size", 2),
            t_patch_size=sr_dp_config.get("t_patch_size", 1),
            frame_receptive_field=sr_dp_config.get("frame_receptive_field", 11),
            spatial_rope_interpolation=sr_dp_config.get("spatial_rope_interpolation", "extra"),
            ref_audio_offset=sr_dp_config.get("ref_audio_offset", 1000),
            text_offset=sr_dp_config.get("text_offset", 0),
            coords_style="v1",
        )

        # ---- Noise schedule for SR ----
        self.sigmas = ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)

        # ---- Video processor ----
        self.video_processor = VideoProcessor(vae_scale_factor=16)

        # ---- Negative prompt embedding cache ----
        self._context_null = None
        self._original_context_null_len = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    # ---- Text encoding ----
    @torch.inference_mode()
    def encode_prompt(self, prompt: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
        """Encode text prompt using T5-Gemma and pad/trim to target length."""
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("Text encoder not loaded. Check model path.")
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        outputs = self.text_encoder(**inputs)
        txt_feat = outputs["last_hidden_state"].to(dtype)
        txt_feat, original_len = pad_or_trim(txt_feat, target_size=self.t5_gemma_target_length, dim=1)
        return txt_feat.to(torch.float32), original_len

    def _get_negative_prompt_embedding(self, device, dtype):
        """Get cached negative prompt embedding."""
        if self._context_null is None:
            self._context_null, self._original_context_null_len = self.encode_prompt(
                _DEFAULT_NEGATIVE_PROMPT, device, dtype
            )
        return self._context_null, self._original_context_null_len

    # ---- Image encoding ----
    def encode_image(self, image: PIL.Image.Image, height: int, width: int) -> torch.Tensor:
        """Encode image to VAE latent space."""
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")
        image = _resizecrop(image, height, width)
        image_tensor = self.video_processor.preprocess(image, height=height, width=width)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.bfloat16).unsqueeze(2)
        latent = self.vae.encode(image_tensor)
        # Handle different VAE output formats
        if hasattr(latent, "latent_dist"):
            latent = latent.latent_dist.mode()
        elif hasattr(latent, "latents"):
            latent = latent.latents
        return latent.to(torch.float32)

    # ---- DiT forward wrapper ----
    def _dit_forward(self, eval_input: EvalInput, use_sr_model: bool = False):
        if use_sr_model and self.sr_dit is not None:
            processed = self.sr_data_proxy.process_input(eval_input)
            noise_pred = self.sr_dit(*processed)
            return self.sr_data_proxy.process_output(noise_pred)
        else:
            processed = self.data_proxy.process_input(eval_input)
            noise_pred = self.dit(*processed)
            return self.data_proxy.process_output(noise_pred)

    # ---- Denoising loop ----
    @torch.inference_mode()
    def _evaluate_with_latent(
        self,
        context: torch.Tensor,
        original_context_len: int,
        latent_image: torch.Tensor | None,
        latent_video: torch.Tensor,
        latent_audio: torch.Tensor,
        num_inference_steps: int,
        is_a2v: bool = False,
        use_sr_model: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=self.shift)
        audio_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=self.shift)
        timesteps = video_scheduler.timesteps

        latent_length = latent_video.shape[2]
        sr_video_txt_guidance_scale = (
            torch.tensor(self.sr_video_txt_guidance_scale, device=self.device).expand(1, 1, latent_length, 1, 1).clone()
        )
        if self.use_cfg_trick:
            sr_video_txt_guidance_scale[:, :, : self.cfg_trick_start_frame] = min(
                self.cfg_trick_value, self.sr_video_txt_guidance_scale
            )

        context_null, original_context_null_len = self._get_negative_prompt_embedding(self.device, context.dtype)

        for idx, t in enumerate(timesteps):
            if latent_image is not None:
                latent_video[:, :, :1] = latent_image[:, :, :1]

            video_txt_guidance_scale = self.video_txt_guidance_scale if t > 500 else 2.0

            # Conditional forward
            eval_input_cond = EvalInput(
                x_t=latent_video,
                audio_x_t=latent_audio,
                audio_feat_len=[latent_audio.shape[1]],
                txt_feat=context,
                txt_feat_len=[original_context_len],
            )
            v_output = self._dit_forward(eval_input_cond, use_sr_model=use_sr_model)
            v_cond_video = v_output[0]
            v_cond_audio = v_output[1]

            cfg_number = self.sr_cfg_number if use_sr_model else self.cfg_number
            if cfg_number == 1:
                v_cfg_video = v_cond_video
                v_cfg_audio = v_cond_audio
            elif cfg_number == 2:
                # Unconditional forward
                eval_input_uncond = EvalInput(
                    x_t=latent_video,
                    audio_x_t=latent_audio,
                    audio_feat_len=[latent_audio.shape[1]],
                    txt_feat=context_null,
                    txt_feat_len=[original_context_null_len],
                )
                v_output_uncond = self._dit_forward(eval_input_uncond, use_sr_model=use_sr_model)
                v_uncond_video = v_output_uncond[0]
                v_uncond_audio = v_output_uncond[1]
                if use_sr_model:
                    v_cfg_video = v_uncond_video + sr_video_txt_guidance_scale * (v_cond_video - v_uncond_video)
                else:
                    v_cfg_video = v_uncond_video + video_txt_guidance_scale * (v_cond_video - v_uncond_video)
                v_cfg_audio = v_uncond_audio + self.audio_txt_guidance_scale * (v_cond_audio - v_uncond_audio)
            else:
                raise ValueError(f"Invalid cfg_number: {cfg_number}")

            latent_video, latent_audio = schedule_latent_step(
                video_scheduler=video_scheduler,
                audio_scheduler=audio_scheduler,
                latent_video=latent_video,
                latent_audio=latent_audio,
                t=t,
                idx=idx,
                steps=num_inference_steps,
                v_cfg_video=v_cfg_video,
                v_cfg_audio=v_cfg_audio,
                is_a2v=is_a2v,
                cfg_number=cfg_number,
                use_sr_model=use_sr_model,
                using_sde_flag=self.using_sde_flag,
            )

        if latent_image is not None:
            latent_video[:, :, :1] = latent_image[:, :, :1]

        return latent_video, latent_audio

    # ---- Video decoding ----
    def _decode_video(self, latent: torch.Tensor) -> list | None:
        if self.vae is None:
            return None
        videos = self.vae.decode(latent.to(self.vae.dtype))
        if hasattr(videos, "sample"):
            videos = videos.sample
        if videos is None:
            return None
        videos = videos.float()
        videos.mul_(0.5).add_(0.5).clamp_(0, 1)
        videos = [video.cpu() for video in videos]
        videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
        videos = [video.numpy().astype(np.uint8) for video in videos]
        return videos

    # ---- Audio decoding ----
    def _decode_audio(self, latent_audio: torch.Tensor) -> np.ndarray | None:
        if self.audio_vae is None:
            return None
        vae_dtype = next(self.audio_vae.vae_model.parameters()).dtype
        audio_output = self.audio_vae.decode(latent_audio.permute(0, 2, 1).to(vae_dtype))
        audio_output_np = audio_output.float().squeeze(0).T.cpu().numpy()
        return audio_output_np

    # ---- Main forward ----
    @torch.inference_mode()
    def forward(
        self,
        req: OmniDiffusionRequest,
        height: int = 480,
        width: int = 832,
        seconds: int = 5,
        num_inference_steps: int = 32,
        sr_height: int | None = None,
        sr_width: int | None = None,
        sr_num_inference_steps: int = 5,
        **kwargs,
    ) -> DiffusionOutput:
        """Generate video + audio from text/image/audio prompt.

        Args:
            req: The diffusion request containing prompts and sampling params.
            height: Base resolution height.
            width: Base resolution width.
            seconds: Duration of generated video in seconds.
            num_inference_steps: Denoising steps for base resolution.
            sr_height: Super-resolution height (None to skip SR).
            sr_width: Super-resolution width (None to skip SR).
            sr_num_inference_steps: Denoising steps for SR.

        Returns:
            DiffusionOutput with video and audio data.
        """
        # ---- Parse request ----
        if len(req.prompts) < 1:
            raise ValueError("At least one prompt is required.")

        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt = prompt_data
            image = None
            audio_path = None
        else:
            prompt = prompt_data.get("prompt", "")
            multi_modal_data = prompt_data.get("multi_modal_data", {})
            image = multi_modal_data.get("image", None)
            audio_path = multi_modal_data.get("audio", None)

        # Override from sampling_params
        sp = req.sampling_params
        height = sp.height or height
        width = sp.width or width
        num_inference_steps = sp.num_inference_steps or num_inference_steps
        if sp.seed is not None:
            torch.manual_seed(sp.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(sp.seed)

        # ---- Compute latent dimensions ----
        br_latent_height = height // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        br_latent_width = width // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        br_height = br_latent_height * self.vae_stride[1]
        br_width = br_latent_width * self.vae_stride[2]

        # ---- Initialize audio latent ----
        if audio_path is not None and self.audio_vae is not None:
            # Load and encode reference audio
            try:
                from .magi_human_audio_utils import load_audio_and_encode

                latent_audio = load_audio_and_encode(self.audio_vae, audio_path, seconds)
                latent_audio = latent_audio.permute(0, 2, 1)
                num_frames = latent_audio.shape[1]
                is_a2v = True
            except ImportError:
                num_frames = seconds * self.fps + 1
                latent_audio = torch.randn(1, num_frames, 64, dtype=torch.float32, device=self.device)
                is_a2v = False
        else:
            num_frames = seconds * self.fps + 1
            latent_audio = torch.randn(1, num_frames, 64, dtype=torch.float32, device=self.device)
            is_a2v = False

        latent_length = (num_frames - 1) // 4 + 1

        # ---- Initialize video latent ----
        latent_video = torch.randn(
            1,
            self.z_dim,
            latent_length,
            br_latent_height,
            br_latent_width,
            dtype=torch.float32,
            device=self.device,
        )

        # ---- Encode text ----
        context, original_context_len = self.encode_prompt(prompt, self.device, torch.bfloat16)

        # ---- Encode image condition ----
        br_image = None
        if image is not None:
            if isinstance(image, str):
                image = PIL.Image.open(image).convert("RGB")
            br_image = self.encode_image(image, br_height, br_width)

        # ---- Base resolution denoising ----
        br_latent_video, br_latent_audio = self._evaluate_with_latent(
            context,
            original_context_len,
            br_image,
            latent_video.clone(),
            latent_audio.clone(),
            num_inference_steps,
            is_a2v,
            use_sr_model=False,
        )

        # ---- Super resolution (optional) ----
        if sr_width is not None and sr_height is not None and self.sr_dit is not None:
            sr_latent_height = sr_height // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
            sr_latent_width = sr_width // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
            sr_height = sr_latent_height * self.vae_stride[1]
            sr_width = sr_latent_width * self.vae_stride[2]

            sr_image = None
            if image is not None:
                sr_image = self.encode_image(image, sr_height, sr_width)

            # Upsample base latent to SR resolution
            latent_video = torch.nn.functional.interpolate(
                br_latent_video,
                size=(latent_length, sr_latent_height, sr_latent_width),
                mode="trilinear",
                align_corners=True,
            )

            # Add noise for SR
            if self.noise_value != 0:
                noise = torch.randn_like(latent_video, device=latent_video.device)
                sigmas = self.sigmas.to(latent_video.device)
                sigma = sigmas[self.noise_value]
                latent_video = latent_video * sigma + noise * (1 - sigma**2) ** 0.5

            latent_audio_sr = torch.randn_like(
                br_latent_audio, device=br_latent_audio.device
            ) * self.sr_audio_noise_scale + br_latent_audio * (1 - self.sr_audio_noise_scale)

            latent_video, _ = self._evaluate_with_latent(
                context,
                original_context_len,
                sr_image,
                latent_video.clone(),
                latent_audio_sr.clone(),
                sr_num_inference_steps,
                is_a2v,
                use_sr_model=True,
            )
            latent_audio = br_latent_audio
        else:
            latent_video = br_latent_video
            latent_audio = br_latent_audio

        # ---- Decode video ----
        torch.cuda.empty_cache()
        videos_np = self._decode_video(latent_video)

        # ---- Decode audio ----
        audio_np = self._decode_audio(latent_audio)

        # ---- Build output ----
        custom_output = {}
        if videos_np is not None:
            custom_output["video"] = videos_np[0] if videos_np else None
        if audio_np is not None:
            custom_output["audio"] = audio_np

        return DiffusionOutput(
            output=custom_output,
            custom_output=custom_output,
        )
