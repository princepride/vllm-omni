# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file is a modified version for Bagel model split:
# Image Generator with MoE layers for generation mode.
# Retains und weights for <|vision_start|> and <|vision_end|> tokens.


import glob
import os
from collections.abc import Iterable
from dataclasses import dataclass
from math import isqrt

import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor, nn
from tqdm import tqdm
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.vllm_flash_attn import flash_attn_varlen_func

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .bagel_core import BagelConfig, BagelGenParams, PositionEmbedding, TimestepEmbedder, prepare_vae_latent
from .qwen2_navit import NaiveCache, Qwen2Config


class Qwen2GenAttention(nn.Module):
    """
    Qwen2 attention with MoE routing for generation mode.
    Retains both und and gen weights for handling:
    - <|vision_start|> / <|vision_end|> tokens: use und weights
    - VAE tokens: use gen (_moe_gen) weights
    """

    def __init__(self, config: Qwen2Config, layer_idx: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Und weights (for text tokens like <|vision_start|>)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Gen weights (for VAE tokens) - _moe_gen suffix
        self.q_proj_moe_gen = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = False,  # Usually False for gen mode
        packed_vae_token_indexes: torch.Tensor | None = None,
        packed_text_indexes: torch.Tensor | None = None,
    ):
        # Gen mode: route text tokens through und weights, VAE tokens through gen weights
        packed_query_sequence = packed_query_sequence.to(torch.bfloat16)

        packed_query_states = packed_query_sequence.new_zeros(
            (packed_query_sequence.shape[0], self.num_heads * self.head_dim)
        )
        packed_key_states = packed_query_sequence.new_zeros(
            (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )
        packed_value_states = packed_query_sequence.new_zeros(
            (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )

        packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
        packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

        # Text tokens (<|vision_start|>, <|vision_end|>) use und weights
        packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
        packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)

        packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
        packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)

        packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
        packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        # Apply QK normalization with routing
        packed_query_states = packed_query_states.to(torch.float32)
        packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
        packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
            packed_query_states[packed_vae_token_indexes]
        )

        packed_key_states = packed_key_states.to(torch.float32)
        packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
        packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])

        # Apply rotary position embeddings
        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        # Merge with past key values from text encoder
        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        # Full attention (non-causal) for generation
        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=False,  # Non-causal for generation
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)

        # Route output projection
        packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
        packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen2GenDecoderLayer(nn.Module):
    """
    Qwen2 decoder layer with MoE routing for generation.
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2GenAttention(config, layer_idx)

        # Und weights for text tokens
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gen weights for VAE tokens
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        packed_text_indexes: torch.Tensor | None = None,
        packed_vae_token_indexes: torch.Tensor | None = None,
        past_key_values: NaiveCache | None = None,
        update_past_key_values: bool = False,
    ) -> tuple[torch.Tensor, NaiveCache | None]:
        residual = packed_query_sequence

        # Input LayerNorm with routing
        packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
        packed_query_sequence_[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
        packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
            packed_query_sequence[packed_vae_token_indexes]
        )
        packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
            past_key_values=past_key_values,
            update_past_key_values=update_past_key_values,
        )
        packed_query_sequence = residual + packed_query_sequence

        # FFN with routing
        residual = packed_query_sequence
        packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
        packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

        packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
        packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(torch.bfloat16)

        packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
        packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
        packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
        packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2ImageGeneratorModel(Qwen2PreTrainedModel):
    """
    Qwen2 model for image generation.
    Contains MoE routing for text vs VAE tokens.
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding for <|vision_start|> and <|vision_end|> tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [Qwen2GenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final norm with routing
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.post_init()

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        packed_text_indexes: torch.Tensor | None = None,
        packed_vae_token_indexes: torch.Tensor | None = None,
        past_key_values: NaiveCache | None = None,
        update_past_key_values: bool = False,
    ) -> tuple[torch.Tensor, NaiveCache | None]:
        # Create position embeddings
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        for decoder_layer in self.layers:
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                packed_text_indexes=packed_text_indexes,
                packed_vae_token_indexes=packed_vae_token_indexes,
                past_key_values=past_key_values,
                update_past_key_values=update_past_key_values,
            )

        # Final norm with routing
        packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
        packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
        packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(
            packed_query_sequence[packed_vae_token_indexes]
        )
        packed_query_sequence = packed_query_sequence_

        return packed_query_sequence, past_key_values


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    downsample: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def default_ae_params() -> AutoEncoderParams:
    return AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )


class Qwen2ImageGenerator(Qwen2PreTrainedModel):
    """
    Qwen2 Image Generator for Bagel.
    Wrapper around Qwen2ImageGeneratorModel for consistent API.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config, bagel_config: BagelConfig = None):
        super().__init__(config)
        self.model = Qwen2ImageGeneratorModel(config)
        self.vocab_size = config.vocab_size

        if bagel_config:
            self.bagel_config = bagel_config
            self.hidden_size = config.hidden_size
            self.latent_patch_size = bagel_config.latent_patch_size
            self.max_latent_size = bagel_config.max_latent_size
            self.latent_channel = bagel_config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel

            # Additional layers for DiT
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

            # Initialize VAE
            self.vae = AutoEncoder(default_ae_params())

            # Init weights for new layers
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = False,
        packed_vae_token_indexes: torch.Tensor | None = None,
        packed_text_indexes: torch.Tensor | None = None,
    ):
        return self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

    @torch.no_grad
    def forward_step(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        **kwargs,
    ):
        """
        Use image_generator for flow matching generation.
        Receives KV cache from text_encoder.
        """
        # Get embeddings for <|vision_start|> and <|vision_end|> tokens
        packed_text_embedding = self.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)

        # Project x_t to LLM dim and add embeddings
        x_t_embed = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed

        if x_t_embed.dtype != packed_sequence.dtype:
            x_t_embed = x_t_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t_embed

        # Forward through image generator model
        output_sequence, _ = self.model(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

        # Project back to VAE dim
        v_t = self.llm2vae(output_sequence)
        v_t = v_t[packed_vae_token_indexes]

        return v_t

    def _decode_image_from_latent(
        self, vae: AutoEncoder, latent: torch.Tensor, image_shape: tuple[int, int]
    ) -> Image.Image:
        H, W = image_shape
        h, w = (
            H // self.bagel_config.vae_config.downsample // self.bagel_config.latent_patch_size,
            W // self.bagel_config.vae_config.downsample // self.bagel_config.latent_patch_size,
        )
        p = self.bagel_config.latent_patch_size
        c = self.bagel_config.vae_config.z_channels
        latent = latent.reshape(1, h, w, p, p, c)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, c, h * p, w * p)

        # Cast to VAE dtype
        vae_dtype = next(vae.parameters()).dtype
        latent = latent.to(vae_dtype)

        image = vae.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(image.to(torch.uint8).cpu().numpy())

    def generate(
        self,
        past_key_values: NaiveCache,
        new_token_ids: dict,
        req: OmniDiffusionRequest = None,
        height: int = 1024,
        width: int = 1024,
    ):
        kv_lens = [past_key_values.key_cache[0].shape[0]]
        ropes = [past_key_values.key_cache[0].shape[0]]
        steps = 50
        if req and req.num_inference_steps:
            steps = int(req.num_inference_steps)

        gen_params = BagelGenParams(
            num_timesteps=steps,
            timestep_shift=3.0,
        )
        generation_input = prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[(height, width)],
            new_token_ids=new_token_ids,
            bagel_config=self.bagel_config,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(self.device)

        with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
            # Generation Loop
            x_t = generation_input["packed_init_noises"]
            timestep_shift = gen_params.timestep_shift
            num_timesteps = gen_params.num_timesteps

            timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
            timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
            dts = timesteps[:-1] - timesteps[1:]
            timesteps = timesteps[:-1]

            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)

                v_t = self.forward_step(x_t=x_t, timestep=timestep, past_key_values=past_key_values, **generation_input)

                x_t = x_t - v_t.to(x_t.device) * dts[i]

            latents = x_t.split((generation_input["packed_seqlens"] - 2).tolist())
        img = self._decode_image_from_latent(self.vae, latents[0], (height, width))
        return DiffusionOutput(output=img)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """
        Load weights with specialized logic for MoE Gen layers.
        Loads standard weights (und) AND _moe_gen weights.
        Also loads time_embedder, vae2llm, llm2vae, latent_pos_embed.
        """
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Map standard Qwen2 keys to our model structure
            if name.startswith("model."):
                name = name[6:]  # remove model. prefix to match Qwen2ImageGeneratorModel structure if needed,
                # but usually we keep it or valid mapping.
                # self.model is Qwen2ImageGeneratorModel
                # Actually, self.named_parameters() will return "model.layers.0..."
                name = "model." + name

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                # Fallback or specific mapping
                pass


def load_bagel_weights(model_path: str, image_generator: Qwen2ImageGenerator, text_encoder: nn.Module = None):
    """
    Helper function to load weights from a checkpoint directory into image_generator and optional text_encoder.
    Handles key mapping and parameter resizing (e.g. latent_pos_embed).
    """
    print(f"Loading weights from {model_path}...")
    full_state_dict = {}

    # Load all safetensors
    chk_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not chk_files:
        print(f"No .safetensors found in {model_path}")
        return

    for bf in chk_files:
        state = load_file(bf)
        for k, v in state.items():
            new_k = k
            if k.startswith("language_model."):
                new_k = k[len("language_model.") :]
            elif k.startswith("vae_model."):
                new_k = "vae." + k[len("vae_model.") :]
            elif (
                k.startswith("encoder.")
                or k.startswith("decoder.")
                or k.startswith("post_quant_conv.")
                or k.startswith("quant_conv.")
            ):
                new_k = "vae." + k

            # Handle latent_pos_embed resize if needed
            if new_k == "latent_pos_embed.pos_embed" and v.shape != image_generator.latent_pos_embed.pos_embed.shape:
                print(f"Resizing latent_pos_embed from {image_generator.latent_pos_embed.pos_embed.shape} to {v.shape}")
                npos, hdim = v.shape
                side = isqrt(int(npos))
                if side * side == int(npos) and hdim == int(image_generator.hidden_size):
                    # Resize model parameter
                    curr_param = image_generator.latent_pos_embed.pos_embed
                    curr_param.data = curr_param.data.new_empty((npos, hdim))
                    # Update configs
                    if hasattr(image_generator, "bagel_config") and image_generator.bagel_config:
                        image_generator.bagel_config.max_latent_size = int(side)
                    image_generator.max_latent_size = int(side)
                    if hasattr(image_generator.latent_pos_embed, "max_num_patch_per_side"):
                        image_generator.latent_pos_embed.max_num_patch_per_side = int(side)

            full_state_dict[new_k] = v

    # Load to text_encoder
    if text_encoder is not None:
        text_encoder_dict = {
            k: v
            for k, v in full_state_dict.items()
            if "_moe_gen" not in k
            and not k.startswith("vae.")
            and not k.startswith("time_embedder")
            and not k.startswith("vae2llm")
            and not k.startswith("llm2vae")
            and not k.startswith("latent_pos_embed")
        }
        print("Loading state dict into text_encoder...")
        text_encoder.load_state_dict(text_encoder_dict, strict=False)

    # Load to image_generator
    print("Loading state dict into image_generator...")
    image_generator.load_state_dict(full_state_dict, strict=False)
