# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Ported from daVinci-MagiHuman inference/pipeline/video_generate.py
# Adapted for vllm-omni: single-GPU, diffusers VAE, configurable dit_subfolder.

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.models.t5gemma import T5GemmaEncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .magi_human_audio_utils import load_audio_and_encode
from .magi_human_audio_vae import SAAudioFeatureExtractor
from .magi_human_data_proxy import MagiDataProxy
from .magi_human_dit import DiTModel, MagiHumanDiTConfig
from .magi_human_scheduler import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalInput – the intermediary structure fed to DiT via MagiDataProxy
# ---------------------------------------------------------------------------
@dataclass
class EvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | list[int]


# ---------------------------------------------------------------------------
# Text encoder wrapper (cached singleton within pipeline)
# ---------------------------------------------------------------------------
class _T5GemmaEncoder:
    def __init__(self, model_path: str, device: str, weight_dtype: torch.dtype):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = T5GemmaEncoderModel.from_pretrained(model_path, is_encoder_decoder=False, dtype=weight_dtype).to(
            device
        )

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs["last_hidden_state"].half()


def _pad_or_trim(tensor: torch.Tensor, target_size: int, dim: int, pad_value: float = 0.0) -> tuple[torch.Tensor, int]:
    current_size = tensor.size(dim)
    if current_size < target_size:
        padding_amount = target_size - current_size
        padding_tuple = [0] * (2 * tensor.dim())
        padding_dim_index = tensor.dim() - 1 - dim
        padding_tuple[2 * padding_dim_index + 1] = padding_amount
        return F.pad(tensor, tuple(padding_tuple), "constant", pad_value), current_size
    slicing = [slice(None)] * tensor.dim()
    slicing[dim] = slice(0, target_size)
    return tensor[tuple(slicing)], target_size


def _get_padded_t5_gemma_embedding(
    prompt: str,
    encoder: _T5GemmaEncoder,
    target_length: int,
) -> tuple[torch.Tensor, int]:
    txt_feat = encoder.encode(prompt)
    txt_feat, original_len = _pad_or_trim(txt_feat, target_size=target_length, dim=1)
    return txt_feat.to(torch.float32), original_len


def _resizecrop(img: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Centre-crop resize keeping aspect ratio then letterbox to target."""
    pil_image = img.convert("RGB")
    original_width, original_height = pil_image.size
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = max(scale_x, scale_y)
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    return resized_image.crop((left, top, left + target_width, top + target_height))


# ---------------------------------------------------------------------------
# Scheduling helper
# ---------------------------------------------------------------------------
def _schedule_latent_step(
    *,
    video_scheduler: FlowUniPCMultistepScheduler,
    audio_scheduler: FlowUniPCMultistepScheduler,
    latent_video: torch.Tensor,
    latent_audio: torch.Tensor,
    t,
    idx: int,
    steps,
    v_cfg_video: torch.Tensor,
    v_cfg_audio: torch.Tensor,
    is_a2v: bool,
    cfg_number: int,
    using_sde_flag: bool,
):
    if cfg_number == 1:
        latent_video = video_scheduler.step_ddim(v_cfg_video, idx, latent_video)
        latent_audio = audio_scheduler.step_ddim(v_cfg_audio, idx, latent_audio)
        return latent_video, latent_audio

    if using_sde_flag:
        if idx < int(len(steps) * (3 / 4)):
            noise_theta = 1.0 if (idx + 1) % 2 == 0 else 0.0
        else:
            noise_theta = 1.0 if idx % 3 == 0 else 0.0
        latent_video = video_scheduler.step_sde(v_cfg_video, idx, latent_video, noise_theta=noise_theta)
        if not is_a2v:
            latent_audio = audio_scheduler.step_sde(v_cfg_audio, idx, latent_audio, noise_theta=noise_theta)
        return latent_video, latent_audio

    latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
    if not is_a2v:
        latent_audio = audio_scheduler.step(v_cfg_audio, t, latent_audio, return_dict=False)[0]
    return latent_video, latent_audio


# ---------------------------------------------------------------------------
# Negative prompt (same as original)
# ---------------------------------------------------------------------------
_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
    ", low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, "
    "compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, "
    "reverb, room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, "
    "grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, "
    "sudden silence, looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound "
    "effects, chaotic, disorganized, messy, cheap sound"
    ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat "
    "intonation, undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, "
    "insincere, fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, "
    "unconfident, tired, weak voice, stuttering, stammering, mumbling, slurred speech, mispronounced, "
    "bad articulation, lisp, vocal fry, creaky voice, mouth clicks, lip smacks, wet mouth sounds, heavy "
    "breathing, audible inhales, plosives, p-pops, coughing, clearing throat, sneezing, speaking too fast, "
    "rushed, speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, disjointed, multiple "
    "speakers, two voices, background talking, out of tune, off-key, autotune artifacts"
)


# ---------------------------------------------------------------------------
# Pre/post process funcs (registered in registry)
# ---------------------------------------------------------------------------
def get_magi_human_pre_process_func(*args, **kwargs):
    def pre_process(request: OmniDiffusionRequest):
        return request

    return pre_process


def get_magi_human_post_process_func(*args, **kwargs):
    def post_process(output: DiffusionOutput) -> DiffusionOutput:
        return output

    return post_process


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
class MagiHumanPipeline(nn.Module, ProgressBarMixin, DiffusionPipelineProfilerMixin):
    def __init__(self, od_config: OmniDiffusionConfig, **kwargs):
        super().__init__()
        model_path = od_config.model
        device = f"cuda:{torch.cuda.current_device()}"
        self.device_str = device
        self.dtype = od_config.dtype or torch.bfloat16

        pipeline_config_path = os.path.join(model_path, "pipeline_config.json")
        with open(pipeline_config_path) as f:
            pipeline_config = json.load(f)
        eval_cfg = pipeline_config["evaluation"]
        dp_cfg = pipeline_config["data_proxy"]

        dit_subfolder = eval_cfg.get("dit_subfolder", "dit")

        dit_config_path = os.path.join(model_path, dit_subfolder, "config.json")
        with open(dit_config_path) as f:
            dit_json = json.load(f)
        dit_model_config = MagiHumanDiTConfig(**dit_json)

        self.dit = DiTModel(dit_model_config)
        self.dit.eval()

        self.vae = DistributedAutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
        self.vae.to(device)
        self.vae.eval()

        self.audio_vae = SAAudioFeatureExtractor(device=device, model_path=os.path.join(model_path, "audio_vae"))

        logger.info("Loading T5Gemma text encoder from %s", os.path.join(model_path, "text_encoder"))
        self.txt_encoder = _T5GemmaEncoder(
            model_path=os.path.join(model_path, "text_encoder"),
            device=device,
            weight_dtype=self.dtype,
        )

        self.data_proxy = MagiDataProxy(
            patch_size=dp_cfg.get("patch_size", 2),
            t_patch_size=dp_cfg.get("t_patch_size", 1),
            frame_receptive_field=dp_cfg.get("frame_receptive_field", 11),
            spatial_rope_interpolation=dp_cfg.get("spatial_rope_interpolation", "extra"),
            ref_audio_offset=dp_cfg.get("ref_audio_offset", 1000),
            text_offset=dp_cfg.get("text_offset", 0),
            coords_style=dp_cfg.get("coords_style", "v2"),
        )

        self.fps = eval_cfg.get("fps", 25)
        self.num_inference_steps_default = eval_cfg.get("num_inference_steps", 32)
        self.video_txt_guidance_scale = eval_cfg.get("video_txt_guidance_scale", 5.0)
        self.audio_txt_guidance_scale = eval_cfg.get("audio_txt_guidance_scale", 5.0)
        self.shift = eval_cfg.get("shift", 5.0)
        self.cfg_number = eval_cfg.get("cfg_number", 2)
        self.use_cfg_trick = eval_cfg.get("use_cfg_trick", True)
        self.cfg_trick_start_frame = eval_cfg.get("cfg_trick_start_frame", 13)
        self.cfg_trick_value = eval_cfg.get("cfg_trick_value", 2.0)
        self.using_sde_flag = eval_cfg.get("using_sde_flag", False)
        self.t5_gemma_target_length = eval_cfg.get("t5_gemma_target_length", 640)
        self.vae_stride = eval_cfg.get("vae_stride", [4, 16, 16])
        self.z_dim = eval_cfg.get("z_dim", 48)
        self.patch_size = eval_cfg.get("patch_size", [1, 2, 2])

        self.context_null, self.original_context_null_len = _get_padded_t5_gemma_embedding(
            _NEGATIVE_PROMPT,
            self.txt_encoder,
            self.t5_gemma_target_length,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=16)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=dit_subfolder,
                revision=None,
                prefix="dit.",
                fall_back_to_pt=True,
            ),
        ]

    # ------------------------------------------------------------------
    # Weight loading (AutoWeightsLoader)
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    # ------------------------------------------------------------------
    # DiT forward pass through data proxy
    # ------------------------------------------------------------------
    def _dit_forward(self, eval_input: EvalInput) -> tuple[torch.Tensor, torch.Tensor]:
        packed = self.data_proxy.process_input(eval_input)
        noise_pred = self.dit(*packed)
        return self.data_proxy.process_output(noise_pred)

    # ------------------------------------------------------------------
    # Denoising loop (from original evaluate_with_latent)
    # ------------------------------------------------------------------
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(num_inference_steps, device=self.device_str, shift=self.shift)
        audio_scheduler.set_timesteps(num_inference_steps, device=self.device_str, shift=self.shift)
        timesteps = video_scheduler.timesteps

        latent_length = latent_video.shape[2]
        sr_video_txt_guidance_scale = (
            torch.tensor(self.video_txt_guidance_scale, device=self.device_str)
            .expand(1, 1, latent_length, 1, 1)
            .clone()
        )
        if self.use_cfg_trick:
            sr_video_txt_guidance_scale[:, :, : self.cfg_trick_start_frame] = min(
                self.cfg_trick_value, self.video_txt_guidance_scale
            )

        with self.progress_bar(total=len(timesteps)) as pbar:
            for idx, t in enumerate(timesteps):
                if latent_image is not None:
                    latent_video[:, :, :1] = latent_image[:, :, :1]

                video_txt_guidance_scale = self.video_txt_guidance_scale if t > 500 else 2.0

                eval_input_cond = EvalInput(
                    x_t=latent_video,
                    audio_x_t=latent_audio,
                    audio_feat_len=[latent_audio.shape[1]],
                    txt_feat=context,
                    txt_feat_len=[original_context_len],
                )

                v_cond_video, v_cond_audio = self._dit_forward(eval_input_cond)

                if self.cfg_number == 1:
                    v_cfg_video = v_cond_video
                    v_cfg_audio = v_cond_audio
                elif self.cfg_number == 2:
                    eval_input_uncond = EvalInput(
                        x_t=latent_video,
                        audio_x_t=latent_audio,
                        audio_feat_len=[latent_audio.shape[1]],
                        txt_feat=self.context_null,
                        txt_feat_len=[self.original_context_null_len],
                    )
                    v_uncond_video, v_uncond_audio = self._dit_forward(eval_input_uncond)
                    v_cfg_video = v_uncond_video + video_txt_guidance_scale * (v_cond_video - v_uncond_video)
                    v_cfg_audio = v_uncond_audio + self.audio_txt_guidance_scale * (v_cond_audio - v_uncond_audio)
                else:
                    raise ValueError(f"Invalid cfg_number: {self.cfg_number}")

                latent_video, latent_audio = _schedule_latent_step(
                    video_scheduler=video_scheduler,
                    audio_scheduler=audio_scheduler,
                    latent_video=latent_video,
                    latent_audio=latent_audio,
                    t=t,
                    idx=idx,
                    steps=timesteps,
                    v_cfg_video=v_cfg_video,
                    v_cfg_audio=v_cfg_audio,
                    is_a2v=is_a2v,
                    cfg_number=self.cfg_number,
                    using_sde_flag=self.using_sde_flag,
                )

                pbar.update()

        if latent_image is not None:
            latent_video[:, :, :1] = latent_image[:, :, :1]
        return latent_video, latent_audio

    # ------------------------------------------------------------------
    # Image encoding (via diffusers VAE)
    # ------------------------------------------------------------------
    def _encode_image(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        image = load_image(image)
        image = _resizecrop(image, height, width)
        image = self.video_processor.preprocess(image, height=height, width=width)
        image = image.to(device=self.device_str, dtype=self.dtype).unsqueeze(2)
        vae_out = self.vae.encode(image)
        if hasattr(vae_out, "latent_dist"):
            return vae_out.latent_dist.mode().to(torch.float32)
        return vae_out.to(torch.float32)

    # ------------------------------------------------------------------
    # Decode video latents → numpy
    # ------------------------------------------------------------------
    def _decode_video(self, latent: torch.Tensor) -> list[np.ndarray]:
        videos = self.vae.decode(latent.to(self.dtype))
        if hasattr(videos, "sample"):
            videos = videos.sample
        videos.mul_(0.5).add_(0.5).clamp_(0, 1)
        videos = [v.float().cpu().permute(1, 2, 3, 0) * 255 for v in videos]
        return [v.numpy().astype(np.uint8) for v in videos]

    # ------------------------------------------------------------------
    # Decode audio latents → numpy
    # ------------------------------------------------------------------
    def _decode_audio(self, latent_audio: torch.Tensor) -> np.ndarray:
        latent_audio = latent_audio.squeeze(0).to(self.dtype)
        audio_output = self.audio_vae.decode(latent_audio.T)
        audio_np = audio_output.squeeze(0).T.float().cpu().numpy()
        target_len = int(audio_np.shape[0] * 441 / 512)
        from scipy.signal import resample

        return resample(audio_np, target_len)

    # ------------------------------------------------------------------
    # Main forward – called by vllm-omni scheduler
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        height: int = 256,
        width: int = 448,
        num_inference_steps: int | None = None,
        seconds: int = 10,
        seed: int | None = None,
        image_path: str | None = None,
        audio_path: str | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if len(req.prompts) >= 1:
            p = req.prompts[0]
            prompt = p if isinstance(p, str) else p.get("prompt", prompt)
            if not isinstance(p, str):
                image_path = p.get("image_path", image_path)
                audio_path = p.get("audio_path", audio_path)
        if prompt is None:
            raise ValueError("prompt is required")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        seed = req.sampling_params.seed if req.sampling_params.seed is not None else seed
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps or self.num_inference_steps_default
        if hasattr(req.sampling_params, "extra_args") and req.sampling_params.extra_args:
            seconds = req.sampling_params.extra_args.get("seconds", seconds)
            audio_path = req.sampling_params.extra_args.get("audio_path", audio_path)
            image_path = req.sampling_params.extra_args.get("image_path", image_path)

        device = self.device_str

        br_latent_height = height // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        br_latent_width = width // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        br_height = br_latent_height * self.vae_stride[1]
        br_width = br_latent_width * self.vae_stride[2]

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if audio_path is not None:
            latent_audio = load_audio_and_encode(self.audio_vae, audio_path, seconds)
            latent_audio = latent_audio.permute(0, 2, 1)
            num_frames = latent_audio.shape[1]
            is_a2v = True
        else:
            num_frames = seconds * self.fps + 1
            latent_audio = torch.randn(1, num_frames, 64, dtype=torch.float32, device=device)
            is_a2v = False

        latent_length = (num_frames - 1) // 4 + 1
        latent_video = torch.randn(
            1,
            self.z_dim,
            latent_length,
            br_latent_height,
            br_latent_width,
            dtype=torch.float32,
            device=device,
        )

        context, original_context_len = _get_padded_t5_gemma_embedding(
            prompt,
            self.txt_encoder,
            self.t5_gemma_target_length,
        )

        if image_path is not None:
            br_image = self._encode_image(load_image(image_path), br_height, br_width)
        else:
            br_image = None

        latent_video, latent_audio = self._evaluate_with_latent(
            context,
            original_context_len,
            br_image,
            latent_video.clone(),
            latent_audio.clone(),
            num_steps,
            is_a2v,
        )

        torch.cuda.empty_cache()
        videos_np = self._decode_video(latent_video)
        torch.cuda.empty_cache()
        audio_np = self._decode_audio(latent_audio)

        return DiffusionOutput(
            output=[],
            custom_output={
                "video": videos_np[0],
                "audio": audio_np,
            },
        )
