# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-U1 Pipeline for vLLM-Omni.

SenseNova-U1 is a unified Qwen3-based model that uses Mixture-of-Tokenizers
(MoT) attention for text-to-image generation via flow matching in patch space.
It has no separate VAE or text encoder — the Qwen3 LLM itself serves as both
the text encoder (via KV cache) and the denoising backbone (via MoT branches).

Key integration points:
- Transformer layers ported with TP support (QKVParallelLinear,
  MergedColumnParallelLinear, RowParallelLinear) in sensenova_u1_transformer.py.
- Vision model (NEOVisionModel) and FM modules kept as standard nn.Module
  since they are lightweight (no transformer blocks).
- Weight loading uses stacked_params_mapping for fused QKV and gate_up.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .sensenova_u1_transformer import (
    SenseNovaU1ForCausalLM,
    clear_flash_kv_cache,
    create_block_causal_mask,
    prepare_flash_kv_cache,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)


# ---------------------------------------------------------------------------
# Simple config namespace parsed from config.json
# ---------------------------------------------------------------------------


class _ConfigNamespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HuggingFace model ID or local path to a local directory."""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def _load_sensenova_u1_config(model_path: str):
    """Parse config.json and return (top_config, llm_config, vision_config) namespaces."""
    local_path = _resolve_model_path(model_path)
    cfg_path = os.path.join(local_path, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)

    llm_raw = raw.get("llm_config", {})
    vis_raw = raw.get("vision_config", {})

    # LLM config needs these extra fields for 3D RoPE
    llm_cfg = _ConfigNamespace(
        hidden_size=llm_raw.get("hidden_size", 4096),
        intermediate_size=llm_raw.get("intermediate_size", 11008),
        num_hidden_layers=llm_raw.get("num_hidden_layers", 32),
        num_attention_heads=llm_raw.get("num_attention_heads", 32),
        num_key_value_heads=llm_raw.get("num_key_value_heads", llm_raw.get("num_attention_heads", 32)),
        head_dim=llm_raw.get("head_dim", llm_raw.get("hidden_size", 4096) // llm_raw.get("num_attention_heads", 32)),
        hidden_act=llm_raw.get("hidden_act", "silu"),
        rms_norm_eps=llm_raw.get("rms_norm_eps", 1e-6),
        vocab_size=llm_raw.get("vocab_size", 152064),
        rope_theta=llm_raw.get("rope_theta", 1000000.0),
        rope_theta_hw=llm_raw.get("rope_theta_hw", 10000.0),
        max_position_embeddings=llm_raw.get("max_position_embeddings", 40960),
        max_position_embeddings_hw=llm_raw.get("max_position_embeddings_hw", 10000),
        attention_bias=llm_raw.get("attention_bias", True),
        layer_types=llm_raw.get("layer_types", ["full_attention"] * llm_raw.get("num_hidden_layers", 32)),
        tie_word_embeddings=llm_raw.get("tie_word_embeddings", False),
    )

    vis_cfg = _ConfigNamespace(
        num_channels=vis_raw.get("num_channels", 3),
        patch_size=vis_raw.get("patch_size", 16),
        hidden_size=vis_raw.get("hidden_size", 1024),
        llm_hidden_size=vis_raw.get("llm_hidden_size", [llm_cfg.hidden_size]),
        downsample_ratio=vis_raw.get("downsample_ratio", [0.5]),
        rope_theta_vision=vis_raw.get("rope_theta_vision", 10000.0),
        max_position_embeddings_vision=vis_raw.get("max_position_embeddings_vision", 10000),
        output_hidden_states=vis_raw.get("output_hidden_states", False),
        use_return_dict=vis_raw.get("use_return_dict", True),
    )

    top_cfg = _ConfigNamespace(
        downsample_ratio=raw.get("downsample_ratio", 0.5),
        template=raw.get("template", "neo1_0"),
        fm_head_layers=raw.get("fm_head_layers", 2),
        fm_head_dim=raw.get("fm_head_dim", 1536),
        fm_head_mlp_ratio=raw.get("fm_head_mlp_ratio", 1.0),
        use_pixel_head=raw.get("use_pixel_head", False),
        noise_scale=raw.get("noise_scale", 1.0),
        noise_scale_mode=raw.get("noise_scale_mode", "none"),
        noise_scale_base_image_seq_len=raw.get("noise_scale_base_image_seq_len", 256),
        noise_scale_max_value=raw.get("noise_scale_max_value", 10.0),
        add_noise_scale_embedding=raw.get("add_noise_scale_embedding", False),
        time_schedule=raw.get("time_schedule", "standard"),
        time_shift_type=raw.get("time_shift_type", "exponential"),
        base_shift=raw.get("base_shift", 0.5),
        max_shift=raw.get("max_shift", 1.15),
        base_image_seq_len=raw.get("base_image_seq_len", 256),
        max_image_seq_len=raw.get("max_image_seq_len", 4096),
        concat_time_token_num=raw.get("concat_time_token_num", 0),
        t_eps=raw.get("t_eps", 0.02),
    )

    return top_cfg, llm_cfg, vis_cfg


# ---------------------------------------------------------------------------
# Vision Embedding — 2D RoPE + Conv patch embed (no transformer blocks)
# ---------------------------------------------------------------------------


def _precompute_rope_freqs_sincos(dim, max_position, base=10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rotary_emb_1d(x, cos_cached, sin_cached, positions):
    cos = cos_cached[positions]
    sin = sin_cached[positions]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _build_abs_positions_from_grid_hw(grid_hw, device=None):
    device = grid_hw.device
    B = grid_hw.shape[0]
    H, W = grid_hw[:, 0], grid_hw[:, 1]
    N = H * W
    N_total = N.sum()
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)
    pid = torch.arange(N_total, device=device)
    pid = pid - torch.cumsum(torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0)[patch_to_sample]
    W_per_patch = W[patch_to_sample]
    return pid % W_per_patch, pid // W_per_patch


class NEOVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        llm_hidden = config.llm_hidden_size
        if isinstance(llm_hidden, (list, tuple)):
            llm_hidden = llm_hidden[0]
        self.llm_embed_dim = llm_hidden
        ds_ratio = config.downsample_ratio
        if isinstance(ds_ratio, (list, tuple)):
            ds_ratio = ds_ratio[0]
        self.downsample_factor = int(1 / ds_ratio)
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            config.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.dense_embedding = nn.Conv2d(
            self.embed_dim, self.llm_embed_dim, kernel_size=self.downsample_factor, stride=self.downsample_factor
        )
        self.gelu = nn.GELU()

        rope_dim = self.embed_dim // 2
        cos_x, sin_x = _precompute_rope_freqs_sincos(
            rope_dim, config.max_position_embeddings_vision, base=config.rope_theta_vision
        )
        cos_y, sin_y = _precompute_rope_freqs_sincos(
            rope_dim, config.max_position_embeddings_vision, base=config.rope_theta_vision
        )
        self.register_buffer("cos_cached_x", cos_x, persistent=False)
        self.register_buffer("sin_cached_x", sin_x, persistent=False)
        self.register_buffer("cos_cached_y", cos_y, persistent=False)
        self.register_buffer("sin_cached_y", sin_y, persistent=False)

    def forward(self, pixel_values, grid_hw=None):
        pixel_values = pixel_values.view(-1, 3, self.patch_size, self.patch_size)
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(-1, self.embed_dim)

        # 2D RoPE
        abs_x, abs_y = _build_abs_positions_from_grid_hw(grid_hw)
        dim_half = self.embed_dim // 2
        p1 = _apply_rotary_emb_1d(
            patch_embeds[:, :dim_half].float(),
            self.cos_cached_x.to(patch_embeds.device),
            self.sin_cached_x.to(patch_embeds.device),
            abs_x,
        )
        p2 = _apply_rotary_emb_1d(
            patch_embeds[:, dim_half:].float(),
            self.cos_cached_y.to(patch_embeds.device),
            self.sin_cached_y.to(patch_embeds.device),
            abs_y,
        )
        patch_embeds = torch.cat([p1, p2], dim=-1).to(self.patch_embedding.weight.dtype)

        patches_list = []
        cur = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            pe = patch_embeds[cur : cur + h * w].view(h, w, -1).unsqueeze(0)
            pe = self.dense_embedding(pe.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            patches_list.append(pe.view(-1, pe.shape[-1]))
            cur += h * w

        return torch.cat(patches_list, dim=0)


class NEOVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = NEOVisionEmbeddings(config)

    def forward(self, pixel_values=None, grid_hw=None, **_kwargs):
        return self.embeddings(pixel_values, grid_hw=grid_hw)


# ---------------------------------------------------------------------------
# FM modules
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000.0):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class ConvDecoder(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024):
        super().__init__()
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(hidden_dim // 4, 192, kernel_size=3, padding=1)
        self.ps3 = nn.PixelShuffle(8)

    def forward(self, x):
        x = self.act1(self.conv1(self.ps1(x)))
        x = self.ps3(self.conv2(self.ps2(x)))
        return x


# ---------------------------------------------------------------------------
# Conversation template (inlined from fastchat MPT style)
# ---------------------------------------------------------------------------


def _build_t2i_query(prompt_text, system_message=None, append_text=None):
    sys_msg = system_message or ""
    system_prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n" if sys_msg else ""
    query = system_prompt + f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    if append_text is not None:
        query += append_text
    return query


# ---------------------------------------------------------------------------
# Patchify / Unpatchify
# ---------------------------------------------------------------------------


def _patchify(images, patch_size, channel_first=False):
    h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
    x = images.reshape(images.shape[0], 3, h, patch_size, w, patch_size)
    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(images.shape[0], h * w, patch_size**2 * 3)


def _unpatchify(x, patch_size, h=None, w=None):
    if h is None or w is None:
        h = w = int(x.shape[1] ** 0.5)
    else:
        h = h // patch_size
        w = w // patch_size
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)


# ---------------------------------------------------------------------------
# Image denormalization for output
# ---------------------------------------------------------------------------


def _denorm(x):
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch):
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def get_sensenova_u1_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(x):
        return x

    return post_process_func


# ---------------------------------------------------------------------------
# CFG helpers
# ---------------------------------------------------------------------------


@torch.amp.autocast("cuda", dtype=torch.float32)
def _optimized_scale(positive_flat, negative_flat):
    dot = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    sq_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot / sq_norm


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SenseNovaU1Pipeline(nn.Module):
    """SenseNova-U1 text-to-image pipeline for vllm-omni.

    Builds the full model graph internally:
    - language_model: SenseNovaU1ForCausalLM (TP-aware)
    - vision_model: NEOVisionModel (understanding branch)
    - fm_modules: ModuleDict with vision_model_mot_gen, timestep_embedder, fm_head, etc.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        model_path = od_config.model
        self.local_model_path = _resolve_model_path(model_path)

        self.top_cfg, self.llm_cfg, self.vis_cfg = _load_sensenova_u1_config(model_path)

        # Tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)

        # Language model (TP-aware)
        self.language_model = SenseNovaU1ForCausalLM(
            self.llm_cfg,
            prefix="language_model",
        )

        # Vision model (understanding branch)
        self.vision_model = NEOVisionModel(self.vis_cfg)

        # FM modules
        llm_hidden_size = self.llm_cfg.hidden_size
        patch_size = self.vis_cfg.patch_size
        merge_size = int(1 / self.top_cfg.downsample_ratio)
        output_dim = 3 * (patch_size * merge_size) ** 2

        vision_model_mot_gen = NEOVisionModel(self.vis_cfg)
        timestep_embedder = TimestepEmbedder(llm_hidden_size)

        if self.top_cfg.use_pixel_head:
            fm_head = ConvDecoder(llm_hidden_size)
        elif self.top_cfg.fm_head_layers > 2:
            fm_head = nn.Sequential(
                nn.Linear(llm_hidden_size, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, output_dim, bias=True),
            )
        else:
            fm_head = nn.Sequential(
                nn.Linear(llm_hidden_size, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, output_dim, bias=True),
            )

        self.fm_modules = nn.ModuleDict(
            {
                "vision_model_mot_gen": vision_model_mot_gen,
                "timestep_embedder": timestep_embedder,
                "fm_head": fm_head,
            }
        )

        if self.top_cfg.add_noise_scale_embedding:
            self.fm_modules["noise_scale_embedder"] = TimestepEmbedder(llm_hidden_size)

        # Config shortcuts
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.downsample_ratio = self.top_cfg.downsample_ratio

        # Weight sources for diffusers_loader
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.local_model_path,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
            ),
        ]

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        if gen_model:
            return self.fm_modules["vision_model_mot_gen"](pixel_values=pixel_values, grid_hw=grid_hw)
        return self.vision_model(pixel_values=pixel_values, grid_hw=grid_hw)

    def _build_t2i_text_inputs(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        t_idx = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=self.device)
        h_idx = torch.zeros_like(t_idx)
        w_idx = torch.zeros_like(t_idx)
        indexes = torch.stack([t_idx, h_idx, w_idx], dim=0)
        attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}
        return input_ids, indexes, attention_mask

    def _build_t2i_image_indexes(self, token_h, token_w, text_len, device):
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)

    def _t2i_prefix_forward(self, input_ids, indexes, attention_mask):
        out = self.language_model.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return out.past_key_values, out.last_hidden_state

    def _t2i_predict_v(
        self, input_embeds, indexes_image, attn_mask, past_key_values, t, z, image_token_num, image_size=None, **_kw
    ):
        B, L = z.shape[0], z.shape[1]
        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            image_gen_indicators=torch.ones(
                (input_embeds.shape[0], input_embeds.shape[1]), dtype=torch.bool, device=input_embeds.device
            ),
            indexes=indexes_image,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )

        if self.top_cfg.use_pixel_head:
            merge_size = self.merge_size
            token_h = image_size[1] // (self.patch_size * merge_size)
            token_w = image_size[0] // (self.patch_size * merge_size)
            img_reshaped = outputs.last_hidden_state[:, -image_token_num:].view(B, token_h, token_w, -1)
            img_2d = torch.einsum("b h w c -> b c h w", img_reshaped).contiguous()
            smoothed_img_2d = self.fm_modules["fm_head"](img_2d)
            smoothed_reshaped = smoothed_img_2d.view(
                B, 3, token_h, self.patch_size * merge_size, token_w, self.patch_size * merge_size
            )
            smoothed_reshaped = torch.einsum("b c h p w q -> b h w p q c", smoothed_reshaped)
            x_pred = smoothed_reshaped.contiguous().view(
                B, L, self.patch_size * merge_size * self.patch_size * merge_size * 3
            )
        else:
            x_pred = self.fm_modules["fm_head"](outputs.last_hidden_state[:, -image_token_num:].view(B, L, -1)).view(
                B, L, -1
            )

        v_pred = (x_pred - z) / (1 - t).clamp_min(self.top_cfg.t_eps)
        return v_pred

    def _apply_time_schedule(self, t, image_seq_len, timestep_shift):
        sigma = 1 - t
        shift = timestep_shift
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
        return 1 - sigma

    def _generate_think(self, prefix_outputs, past_key_values, t_idx, max_think_tokens=1024):
        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        think_end_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        think_token_ids = []
        next_token = torch.argmax(prefix_outputs.logits[:, -1, :], dim=-1)

        for _ in range(max_think_tokens):
            token_item = next_token.item()
            if token_item == eos_token_id:
                break
            if token_item == think_end_token_id:
                self.language_model.model.current_index = t_idx
                outputs = self.language_model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                t_idx += 1
                think_token_ids.append(token_item)
                break

            think_token_ids.append(token_item)
            self.language_model.model.current_index = t_idx
            outputs = self.language_model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            t_idx += 1
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        # Append "\n\n<img>" tokens to cache
        append_ids = self.tokenizer(
            "\n\n<img>",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        t_idx = self._append_text_tokens_to_cache(past_key_values, t_idx, append_ids)

        think_text = self.tokenizer.decode(think_token_ids, skip_special_tokens=False)
        return past_key_values, t_idx, think_text

    def _append_text_tokens_to_cache(self, cache, t_idx, input_ids):
        if input_ids.shape[1] == 0:
            return t_idx
        seq_len = input_ids.shape[1]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        t_indexes = torch.arange(t_idx + 1, t_idx + 1 + seq_len, dtype=torch.long, device=self.device)
        h_indexes = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        w_indexes = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

        past_len = cache.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=self.device)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        causal_mask = torch.where(causal_mask == 1, 0.0, float("-inf"))
        mask[:, :, :, past_len:] = causal_mask
        attention_mask_dict = {"full_attention": mask}

        self.language_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask_dict,
            past_key_values=cache,
            use_cache=True,
        )
        return t_idx + seq_len

    # -----------------------------------------------------------------------
    # Main forward (T2I generation)
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")

        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        height = int(req.sampling_params.height) if req.sampling_params.height else 2048
        width = int(req.sampling_params.width) if req.sampling_params.width else 2048
        image_size = (width, height)

        num_steps = int(req.sampling_params.num_inference_steps or 50)
        cfg_scale = float(extra_args.get("cfg_scale", 4.0))
        cfg_norm = str(extra_args.get("cfg_norm", "none"))
        timestep_shift = float(extra_args.get("timestep_shift", 3.0))
        cfg_interval = tuple(extra_args.get("cfg_interval", (0.0, 1.0)))
        batch_size = int(extra_args.get("batch_size", 1))
        seed = int(req.sampling_params.seed) if req.sampling_params.seed is not None else 42
        think_mode = bool(extra_args.get("think", False))
        t_eps = float(extra_args.get("t_eps", 0.02))
        self.top_cfg.t_eps = t_eps

        merge_size = self.merge_size
        IMG_START_TOKEN = "<img>"

        think_content = "<think>\n" if think_mode else "<think>\n\n</think>\n\n" + IMG_START_TOKEN
        query_condition = _build_t2i_query(prompt, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
        query_uncondition = _build_t2i_query("", append_text=IMG_START_TOKEN)

        input_ids_cond, indexes_cond, mask_cond = self._build_t2i_text_inputs(query_condition)
        input_ids_uncond, indexes_uncond, mask_uncond = self._build_t2i_text_inputs(query_uncondition)

        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        indexes_image_cond = self._build_t2i_image_indexes(
            token_h,
            token_w,
            indexes_cond.shape[1],
            device=self.device,
        )
        indexes_image_uncond = self._build_t2i_image_indexes(
            token_h,
            token_w,
            indexes_uncond.shape[1],
            device=self.device,
        )

        think_text = ""
        if think_mode:
            outputs_cond = self.language_model(
                input_ids=input_ids_cond,
                indexes=indexes_cond,
                attention_mask=mask_cond,
                use_cache=True,
            )
            past_kv_cond = outputs_cond.past_key_values
            t_index_cond = indexes_cond[0].max().item()
            past_kv_cond, t_index_cond, think_text = self._generate_think(
                outputs_cond,
                past_kv_cond,
                t_index_cond,
            )
            indexes_image_cond = self._build_t2i_image_indexes(
                token_h,
                token_w,
                t_index_cond + 1,
                device=self.device,
            )
            hidden_cond = None
        else:
            past_kv_cond, hidden_cond = self._t2i_prefix_forward(input_ids_cond, indexes_cond, mask_cond)

        past_kv_uncond, hidden_uncond = self._t2i_prefix_forward(input_ids_uncond, indexes_uncond, mask_uncond)

        device = self.device
        dtype = self.od_config.dtype

        # Expand cache for batch
        for layer_idx in range(len(past_kv_cond.layers)):
            past_kv_cond.layers[layer_idx].keys = past_kv_cond.layers[layer_idx].keys.expand(
                batch_size, *past_kv_cond.layers[layer_idx].keys.shape[1:]
            )
            past_kv_cond.layers[layer_idx].values = past_kv_cond.layers[layer_idx].values.expand(
                batch_size, *past_kv_cond.layers[layer_idx].values.shape[1:]
            )
            past_kv_uncond.layers[layer_idx].keys = past_kv_uncond.layers[layer_idx].keys.expand(
                batch_size, *past_kv_uncond.layers[layer_idx].keys.shape[1:]
            )
            past_kv_uncond.layers[layer_idx].values = past_kv_uncond.layers[layer_idx].values.expand(
                batch_size, *past_kv_uncond.layers[layer_idx].values.shape[1:]
            )

        prepare_flash_kv_cache(past_kv_cond, current_len=token_h * token_w, batch_size=batch_size)
        prepare_flash_kv_cache(past_kv_uncond, current_len=token_h * token_w, batch_size=batch_size)

        # Init noise
        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        grid_hw = torch.tensor([[grid_h, grid_w]] * batch_size, device=device)

        noise_scale = self.top_cfg.noise_scale
        if self.top_cfg.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.top_cfg.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
            noise_scale = scale * float(self.top_cfg.noise_scale)
            if self.top_cfg.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.top_cfg.noise_scale_max_value)
        generator = torch.Generator(device).manual_seed(seed)
        image_prediction = noise_scale * torch.randn(
            (batch_size, 3, image_size[1], image_size[0]),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        attn_mask_cond = {"full_attention": None}
        attn_mask_uncond = {"full_attention": None}

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        timesteps = self._apply_time_schedule(timesteps, token_h * token_w, timestep_shift)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]

            z = _patchify(image_prediction, self.patch_size * merge_size)
            image_input = _patchify(image_prediction, self.patch_size, channel_first=True)
            image_embeds = self._extract_feature(
                image_input.view(batch_size * grid_h * grid_w, -1),
                gen_model=True,
                grid_hw=grid_hw,
            ).view(batch_size, token_h * token_w, -1)

            t_expanded = t.expand(batch_size * token_h * token_w)
            timestep_embeddings = self.fm_modules["timestep_embedder"](t_expanded).view(
                batch_size, token_h * token_w, -1
            )
            if self.top_cfg.add_noise_scale_embedding:
                ns_tensor = torch.full_like(t_expanded, noise_scale / self.top_cfg.noise_scale_max_value)
                ns_emb = self.fm_modules["noise_scale_embedder"](ns_tensor).view(batch_size, token_h * token_w, -1)
                timestep_embeddings = timestep_embeddings + ns_emb
            image_embeds = image_embeds + timestep_embeddings

            v_pred_cond = self._t2i_predict_v(
                image_embeds,
                indexes_image_cond,
                attn_mask_cond,
                past_kv_cond,
                t,
                z,
                image_token_num=token_h * token_w,
                image_size=image_size,
            )

            if t >= cfg_interval[0] and t <= cfg_interval[1] and cfg_scale > 1:
                v_pred_uncond = self._t2i_predict_v(
                    image_embeds,
                    indexes_image_uncond,
                    attn_mask_uncond,
                    past_kv_uncond,
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    image_size=image_size,
                )
                if cfg_norm == "cfg_zero_star":
                    pos_flat = v_pred_cond.view(batch_size, -1)
                    neg_flat = v_pred_uncond.view(batch_size, -1)
                    alpha = _optimized_scale(pos_flat, neg_flat)
                    alpha = alpha.view(batch_size, *([1] * (len(v_pred_cond.shape) - 1))).to(pos_flat.dtype)
                    if step_i <= 0:
                        v_pred = v_pred_cond * 0.0
                    else:
                        v_pred = v_pred_uncond * alpha + cfg_scale * (v_pred_cond - v_pred_uncond * alpha)
                else:
                    v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
                    if cfg_norm == "global":
                        norm_c = torch.norm(v_pred_cond, dim=(1, 2), keepdim=True)
                        norm_v = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                        v_pred = v_pred * (norm_c / (norm_v + 1e-8)).clamp(0, 1.0)
                    elif cfg_norm == "channel":
                        norm_c = torch.norm(v_pred_cond, dim=-1, keepdim=True)
                        norm_v = torch.norm(v_pred, dim=-1, keepdim=True)
                        v_pred = v_pred * (norm_c / (norm_v + 1e-8)).clamp(0, 1.0)
            else:
                v_pred = v_pred_cond

            z = z + (t_next - t) * v_pred
            image_prediction = _unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

        clear_flash_kv_cache(past_kv_cond)
        clear_flash_kv_cache(past_kv_uncond)

        images = _to_pil(image_prediction)
        img = images[0] if images else None

        custom = {}
        if think_text:
            custom["think_text"] = think_text

        return DiffusionOutput(output=img, custom_output=custom)

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # More specific _mot_gen patterns FIRST to avoid substring
            # ambiguity (e.g. `.q_proj` is a substring of `.q_proj_mot_gen`).
            (".qkv_proj_mot_gen", ".q_proj_mot_gen", "q"),
            (".qkv_proj_mot_gen", ".k_proj_mot_gen", "k"),
            (".qkv_proj_mot_gen", ".v_proj_mot_gen", "v"),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # MLP gate/up fused into MergedColumnParallelLinear
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                if stacked_name not in params_dict:
                    break
                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                loaded = True
                break

            if loaded:
                continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            else:
                assert param.shape == loaded_weight.shape, (
                    f"Shape mismatch for {name}: param={param.shape} vs loaded={loaded_weight.shape}"
                )
                param.data.copy_(loaded_weight)
            loaded_params.add(name)

        return loaded_params
