# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from daVinci-MagiHuman DiT model.
# Copyright (c) 2026 SandAI. All Rights Reserved.

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import Parameter

# ---------------------------------------------------------------------------
# Optional dependencies: magi_compiler & magi_attention
# ---------------------------------------------------------------------------
HAS_MAGI_COMPILER = importlib.util.find_spec("magi_compiler") is not None
HAS_MAGI_ATTENTION = importlib.util.find_spec("magi_attention") is not None
HAS_FA3 = importlib.util.find_spec("flash_attn_interface") is not None
HAS_FA2 = importlib.util.find_spec("flash_attn") is not None


def _is_hopper_arch() -> bool:
    """Check if the current GPU is Hopper architecture (sm_90+)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Local data types (previously from inference.common)
# ---------------------------------------------------------------------------
class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2


@dataclass
class VarlenHandler:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass
class FFAHandler:
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: float | None


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------
class MLPActivationType(Enum):
    SWIGLU7 = "swiglu7"
    GELU7 = "gelu7"


def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: torch.dtype | None = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def gelu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: torch.dtype | None = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu = x.clamp(min=None, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu.to(out_dtype)


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    match activation_type:
        case MLPActivationType.SWIGLU7:
            return swiglu7
        case MLPActivationType.GELU7:
            return gelu7
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")


# ---------------------------------------------------------------------------
# Modality dispatcher
# ---------------------------------------------------------------------------
class ModalityDispatcher:
    permuted_modality_mapping: torch.Tensor
    group_size: torch.Tensor
    group_size_cpu: list[int]
    num_modalities: int

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities
        self.permuted_modality_mapping = self._precompute_permute_mapping(modality_mapping)
        self.group_size = torch.bincount(self.permuted_modality_mapping, minlength=num_modalities).to(torch.int32)
        self.group_size_cpu: list[int] = [int(x) for x in self.group_size.to("cpu").tolist()]

    def _precompute_permute_mapping(self, modality_mapping):
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        return modality_mapping[self.permute_mapping]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        grouped_tensors = torch.split(x, self.group_size_cpu, dim=0)
        return list(grouped_tensors)

    def undispatch(self, *processed_groups: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(processed_groups, dim=0)

    @staticmethod
    def permute(x: torch.Tensor, permute_mapping: torch.Tensor) -> torch.Tensor:
        return x[permute_mapping]

    @staticmethod
    def inv_permute(x: torch.Tensor, inv_permute_mapping: torch.Tensor) -> torch.Tensor:
        return x[inv_permute_mapping]


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------
def freq_bands(
    num_bands: int, temperature: float = 10000.0, step: int = 2, device: torch.device | None = None
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exp)
    return bands


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], dim=-1)


class ElementWiseFourierEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
        learnable: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.in_pixels = in_pixels
        self.learnable = learnable
        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands
        self.device = device
        self.dtype = dtype
        bands = self.get_default_bands()
        if self.learnable:
            self.bands = nn.Parameter(bands)
        else:
            self.bands = nn.Parameter(bands, requires_grad=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_xyz = coords[:, :3]
        sizes = coords[:, 3:6]
        refs = coords[:, 6:9]
        scales = (refs - 1) / (sizes - 1)
        scales[(refs == 1) & (sizes == 1)] = 1
        assert not scales.isnan().any(), "scales has nan"
        assert not scales.isinf().any(), "scales has inf"
        centers = (sizes - 1) / 2
        centers[:, 0] = 0
        coords_xyz = coords_xyz - centers
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        sin_proj = proj.sin()
        cos_proj = proj.cos()
        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)

    def reset_parameters(self):
        bands = self.get_default_bands()
        self.bands.copy_(bands)

    def get_default_bands(self):
        if self.in_pixels:
            raise NotImplementedError("in_pixels are not implemented yet")
        else:
            bands = freq_bands(self.dim // 8, temperature=self.temperature, step=1, device=self.device).to(self.dtype)
        return bands


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class MultiModalityRMSNorm(nn.Module):
    __constants__ = ["dim", "eps", "num_modality"]

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | None = None, num_modality: int = 1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        self.weight = torch.nn.Parameter(torch.zeros(dim * num_modality, device=device, dtype=torch.float32))
        if num_modality > 1:
            self.forward = self.forward_multi_experts
        else:
            self.forward = self.forward_single_expert
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return t

    def forward_multi_experts(self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        original_dtype = x.dtype
        t = self.rms(x)
        weight_chunked = self.weight.chunk(self.num_modality, dim=0)
        t_list = modality_dispatcher.dispatch(t)
        for i in range(self.num_modality):
            t_list[i] = t_list[i] * (weight_chunked[i] + 1)
        t = modality_dispatcher.undispatch(*t_list)
        return t.to(original_dtype)

    def forward_single_expert(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher | None = None
    ) -> torch.Tensor:
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * (self.weight + 1)).to(original_dtype)


# ---------------------------------------------------------------------------
# Linear layers (BF16 compute + MoE per modality)
# ---------------------------------------------------------------------------
class _BF16ComputeLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype | None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        input_cast = input.to(compute_dtype)
        weight_cast = weight.to(compute_dtype)
        output = torch.matmul(input_cast, weight_cast.t())
        if bias is not None:
            bias_cast = bias.to(compute_dtype)
            output = output + bias_cast
        return output.to(output_dtype)


class BaseLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_layers", "num_experts"]

    def __init__(
        self, in_features, out_features, num_layers_for_initialization, num_experts, bias=True, device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": torch.bfloat16}
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers_for_initialization = num_layers_for_initialization
        self.num_experts = num_experts
        self.use_bias = bias
        self.weight = Parameter(torch.empty((out_features * num_experts, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features * num_experts, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input: torch.Tensor,
        output_dtype: torch.dtype | None = None,
        modality_dispatcher: ModalityDispatcher | None = None,
    ) -> torch.Tensor:
        output_dtype = input.dtype if output_dtype is None else output_dtype
        return _BF16ComputeLinear.apply(input, self.weight, self.bias, output_dtype, torch.bfloat16)


class NativeMoELinear(BaseLinear):
    def forward(
        self,
        input: torch.Tensor,
        output_dtype: torch.dtype | None = None,
        modality_dispatcher: ModalityDispatcher | None = None,
    ) -> torch.Tensor:
        output_dtype = input.dtype if output_dtype is None else output_dtype
        input_list = modality_dispatcher.dispatch(input)
        weight_chunked = self.weight.chunk(self.num_experts, dim=0)
        if self.bias is not None:
            bias_chunked = self.bias.chunk(self.num_experts, dim=0)
        for i in range(self.num_experts):
            input_list[i] = _BF16ComputeLinear.apply(
                input_list[i],
                weight_chunked[i],
                bias_chunked[i] if self.bias is not None else None,
                output_dtype,
                torch.bfloat16,
            )
        return modality_dispatcher.undispatch(*input_list)


def create_linear(
    in_features, out_features, num_layers=1, num_experts=1, bias=True, device=None, dtype=None
) -> BaseLinear | NativeMoELinear:
    if num_experts == 1:
        return BaseLinear(in_features, out_features, num_layers, num_experts, bias, device, dtype)
    else:
        return NativeMoELinear(in_features, out_features, num_layers, num_experts, bias, device, dtype)


# ---------------------------------------------------------------------------
# Flash Attention wrappers (no CP support in this version)
# ---------------------------------------------------------------------------
def flash_attn_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Standard flash attention - tries FA3 on Hopper, falls back to FA2."""
    if HAS_FA3 and _is_hopper_arch():
        from flash_attn_interface import flash_attn_func as fa3_flash_attn_func

        return fa3_flash_attn_func(query, key, value)
    elif HAS_FA2:
        from flash_attn.flash_attn_interface import flash_attn_func as fa2_flash_attn_func

        return fa2_flash_attn_func(query, key, value)
    else:
        # Fallback to PyTorch scaled_dot_product_attention
        # query/key/value: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        # Handle GQA: repeat KV heads to match Q heads
        if q.shape[1] != k.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2)


def _split_q_range_with_no_overlap(
    q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> tuple[list[list[int]], list[list[list[int]]]]:
    range_boundary = torch.unique(q_ranges, sorted=True).tolist()
    candidates = [[start, end, []] for start, end in zip(range_boundary[:-1], range_boundary[1:])]
    q_ranges = q_ranges.tolist()
    k_ranges = k_ranges.tolist()
    for q_range, k_range in zip(q_ranges, k_ranges):
        q_start, q_end = q_range
        for q_range_cand in candidates:
            if q_start <= q_range_cand[0] and q_range_cand[1] <= q_end:
                q_range_cand[2].append(k_range)
    q_ranges_out = []
    k_ranges_out = []
    for q_range_cand in candidates:
        if len(q_range_cand[2]) > 0:
            q_ranges_out.append(q_range_cand[0:2])
            k_ranges_out.append(q_range_cand[2])
    return q_ranges_out, k_ranges_out


def _flash_attn_with_correction(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: list[list[int]],
    k_range_list: list[list[list[int]]],
):
    output = torch.zeros_like(query)
    output_lse = torch.zeros((query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device)

    if HAS_FA2:
        from flash_attn.flash_attn_interface import flash_attn_func
    else:
        flash_attn_func = None

    for q_range, k_ranges in zip(q_ranges, k_range_list):
        q_start, q_end = q_range
        qo_out, qo_lse = None, None
        for k_range in k_ranges:
            k_start, k_end = k_range
            if flash_attn_func is not None:
                cur_qo_out, cur_qo_lse, _ = flash_attn_func(
                    query[q_start:q_end].unsqueeze(0),
                    key[k_start:k_end].unsqueeze(0),
                    value[k_start:k_end].unsqueeze(0),
                    return_attn_probs=True,
                )
                cur_qo_out, cur_qo_lse = cur_qo_out.squeeze(0), cur_qo_lse.squeeze(0)
            else:
                # Fallback: scaled_dot_product_attention per chunk
                q_chunk = query[q_start:q_end].unsqueeze(0).transpose(1, 2)
                k_chunk = key[k_start:k_end].unsqueeze(0).transpose(1, 2)
                v_chunk = value[k_start:k_end].unsqueeze(0).transpose(1, 2)
                # Handle GQA: repeat KV heads to match Q heads
                if q_chunk.shape[1] != k_chunk.shape[1]:
                    n_rep = q_chunk.shape[1] // k_chunk.shape[1]
                    k_chunk = k_chunk.repeat_interleave(n_rep, dim=1)
                    v_chunk = v_chunk.repeat_interleave(n_rep, dim=1)
                cur_qo_out = torch.nn.functional.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk)
                cur_qo_out = cur_qo_out.transpose(1, 2).squeeze(0)
                cur_qo_lse = torch.zeros(
                    (cur_qo_out.shape[1], cur_qo_out.shape[0]), dtype=torch.float32, device=query.device
                )

            if qo_out is None:
                qo_out = cur_qo_out
                qo_lse = cur_qo_lse
            else:
                qo_lse[qo_lse == torch.inf] = -torch.inf
                cur_qo_lse[cur_qo_lse == torch.inf] = -torch.inf
                max_lse = torch.max(qo_lse, cur_qo_lse)
                qo_se, cur_qo_se = torch.exp(qo_lse - max_lse), torch.exp(cur_qo_lse - max_lse)
                sum_se = qo_se + cur_qo_se
                qo_scale, cur_qo_scale = qo_se / sum_se, cur_qo_se / sum_se
                qo_out = qo_out * qo_scale.permute(1, 0).unsqueeze(-1) + cur_qo_out * cur_qo_scale.permute(
                    1, 0
                ).unsqueeze(-1)
                qo_lse = torch.log(sum_se) + max_lse

        output[q_start:q_end] = qo_out
        output_lse[q_start:q_end, :] = qo_lse.permute(1, 0)
    return output, output_lse


def flex_flash_attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flexible flash attention with custom Q/K ranges (local attention)."""
    if HAS_MAGI_ATTENTION and _is_hopper_arch():
        from magi_attention.api import flex_flash_attn_func as magi_flex_flash_attn_func

        return magi_flex_flash_attn_func(query, key, value, q_ranges, k_ranges)
    else:
        q_ranges_split, k_range_list = _split_q_range_with_no_overlap(q_ranges, k_ranges)
        return _flash_attn_with_correction(query, key, value, q_ranges_split, k_range_list)


def flash_attn_no_cp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Flash attention without context parallelism."""
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    return flash_attn_func(q, k, v).squeeze(0)


def flex_flash_attn_no_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
) -> torch.Tensor:
    """Flexible flash attention without context parallelism."""
    q, k, v = q.to(torch.bfloat16).squeeze(0), k.to(torch.bfloat16).squeeze(0), v.to(torch.bfloat16).squeeze(0)
    out, _ = flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges)
    return out


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------
@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    params_dtype: torch.dtype
    checkpoint_qk_layernorm_rope: bool
    num_modality: int
    num_layers: int
    use_local_attn: bool = False
    enable_attn_gating: bool = False


class Attention(torch.nn.Module):
    config: AttentionConfig

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, eps=1e-6, num_modality=config.num_modality)
        self.gating_size = config.num_heads_q if config.enable_attn_gating else 0

        self.linear_qkv = create_linear(
            config.hidden_size,
            config.num_heads_q * config.head_dim + config.num_heads_kv * config.head_dim * 2 + self.gating_size,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        self.linear_proj = create_linear(
            config.num_heads_q * config.head_dim,
            config.hidden_size,
            bias=False,
            num_experts=config.num_modality,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        self.q_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)
        self.k_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)

        self.q_size = config.num_heads_q * config.head_dim
        self.kv_size = config.num_heads_kv * config.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler | None,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        qkv: torch.Tensor = self.linear_qkv(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.float32)

        q, k, v, g = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size, self.gating_size], dim=1)
        q = q.view(-1, self.config.num_heads_q, self.config.head_dim)
        k = k.view(-1, self.config.num_heads_kv, self.config.head_dim)
        v = v.view(-1, self.config.num_heads_kv, self.config.head_dim)
        g = g.view(k.shape[0], self.config.num_heads_q, -1)

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        q = ModalityDispatcher.inv_permute(q, inv_permute_mapping).unsqueeze(0)
        k = ModalityDispatcher.inv_permute(k, inv_permute_mapping).unsqueeze(0)
        v = ModalityDispatcher.inv_permute(v, inv_permute_mapping).unsqueeze(0)

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb_torch(q, cos_emb, sin_emb)
        k = apply_rotary_emb_torch(k, cos_emb, sin_emb)

        if self.config.use_local_attn and local_attn_handler is not None:
            self_attn_out = flex_flash_attn_no_cp(q, k, v, local_attn_handler.q_ranges, local_attn_handler.k_ranges)
        else:
            self_attn_out = flash_attn_no_cp(q, k, v)
        self_attn_out = ModalityDispatcher.permute(self_attn_out, permute_mapping)

        if self.config.enable_attn_gating:
            self_attn_out = self_attn_out * torch.sigmoid(g)

        self_attn_out = self_attn_out.view(-1, self.config.num_heads_q * self.config.head_dim).to(torch.bfloat16)
        out = self.linear_proj(self_attn_out, modality_dispatcher=modality_dispatcher)
        return out


# ---------------------------------------------------------------------------
# MLP module
# ---------------------------------------------------------------------------
@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(torch.nn.Module):
    config: MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__()
        num_experts = config.num_modality
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=config.num_modality)
        intermediate_size_up = config.intermediate_size * 2 if config.gated_act else config.intermediate_size

        self.up_gate_proj = create_linear(
            config.hidden_size,
            intermediate_size_up,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=num_experts,
        )
        self.down_proj = create_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=num_experts,
        )
        self.activation_func = create_activation_func(config.activation_type)

    def forward(self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher).to(torch.float32)
        x = self.activation_func(x).to(torch.bfloat16)
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher).to(torch.float32)
        return x


# ---------------------------------------------------------------------------
# Adapter (input embedding)
# ---------------------------------------------------------------------------
@dataclass
class AdapterConfig:
    hidden_size: int
    num_attention_heads: int
    text_in_channels: int
    video_in_channels: int
    audio_in_channels: int
    params_dtype: torch.dtype


class Adapter(torch.nn.Module):
    config: AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.video_embedder = nn.Linear(config.video_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.text_embedder = nn.Linear(config.text_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.audio_embedder = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.rope = ElementWiseFourierEmbed(
            config.hidden_size // config.num_attention_heads, in_pixels=False, learnable=False
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        rope = self.rope(coords_mapping.float())
        embed_dtype = self.video_embedder.weight.dtype
        x = x.to(embed_dtype)
        output_x = torch.zeros(x.shape[0], self.config.hidden_size, device=x.device, dtype=x.dtype)
        output_x[text_mask] = self.text_embedder(x[text_mask, : self.config.text_in_channels])
        output_x[audio_mask] = self.audio_embedder(x[audio_mask, : self.config.audio_in_channels])
        output_x[video_mask] = self.video_embedder(x[video_mask, : self.config.video_in_channels])
        return output_x, rope


# ---------------------------------------------------------------------------
# Transformer layer & block
# ---------------------------------------------------------------------------
class TransFormerLayer(torch.nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        num_modality = 3 if layer_idx in config.mm_layers else 1
        use_local_attn = layer_idx in config.local_attn_layers
        self.post_norm = layer_idx in config.post_norm_layers
        attention_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads_q=config.num_heads_q,
            num_heads_kv=config.num_heads_kv,
            head_dim=config.head_dim,
            params_dtype=config.params_dtype,
            checkpoint_qk_layernorm_rope=config.checkpoint_qk_layernorm_rope,
            num_modality=num_modality,
            num_layers=config.num_layers,
            use_local_attn=use_local_attn,
            enable_attn_gating=config.enable_attn_gating,
        )
        self.attention: Attention = Attention(attention_config)

        activation_type = MLPActivationType.GELU7 if layer_idx in config.gelu7_layers else MLPActivationType.SWIGLU7
        if activation_type == MLPActivationType.SWIGLU7:
            gated_act = True
            intermediate_size = int(config.hidden_size * 4 * 2 / 3) // 4 * 4
        else:
            gated_act = False
            intermediate_size = config.hidden_size * 4
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            params_dtype=config.params_dtype,
            num_modality=num_modality,
            num_layers=config.num_layers,
            gated_act=gated_act,
        )
        self.mlp: MLP = MLP(mlp_config)
        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)
            self.mlp_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler | None,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        attn_out = self.attention(
            hidden_states,
            rope,
            permute_mapping,
            inv_permute_mapping,
            varlen_handler,
            local_attn_handler,
            modality_dispatcher,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + attn_out

        mlp_out = self.mlp(hidden_states, modality_dispatcher)
        if self.post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + mlp_out
        return hidden_states


class TransformerBlock(torch.nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.layers: list[TransFormerLayer] = nn.ModuleList()
        for layer_idx in range(model_config.num_layers):
            self.layers.append(TransFormerLayer(model_config, layer_idx))

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler | None,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                rope,
                permute_mapping,
                inv_permute_mapping,
                varlen_handler,
                local_attn_handler,
                modality_dispatcher,
            )
        return x


# ---------------------------------------------------------------------------
# Main DiT Model
# ---------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    hidden_size: int
    video_in_channels: int
    audio_in_channels: int
    text_in_channels: int
    params_dtype: torch.dtype
    post_process_dtype: torch.dtype


class DiTModel(torch.nn.Module):
    config: TransformerConfig

    def __init__(self, model_config: Any):
        super().__init__()
        self.config = TransformerConfig(
            hidden_size=model_config.hidden_size,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            text_in_channels=model_config.text_in_channels,
            params_dtype=model_config.params_dtype,
            post_process_dtype=torch.float32,
        )
        adapter_config = AdapterConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_heads_q,
            text_in_channels=model_config.text_in_channels,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            params_dtype=torch.float32,
        )
        self.adapter: Adapter = Adapter(adapter_config)
        self.block: TransformerBlock = TransformerBlock(model_config=model_config)
        self.final_norm_video = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_linear_video = nn.Linear(
            self.config.hidden_size, self.config.video_in_channels, bias=False, dtype=torch.float32
        )
        self.final_linear_audio = nn.Linear(
            self.config.hidden_size, self.config.audio_in_channels, bias=False, dtype=torch.float32
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler | None,
    ):
        # No CP dispatch/undispatch in this version
        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        permute_mapping = modality_dispatcher.permute_mapping
        inv_permute_mapping = modality_dispatcher.inv_permute_mapping
        video_mask = modality_mapping == Modality.VIDEO
        audio_mask = modality_mapping == Modality.AUDIO
        text_mask = modality_mapping == Modality.TEXT

        x, rope = self.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
        x = x.to(self.config.params_dtype)
        x = ModalityDispatcher.permute(x, permute_mapping)
        x = self.block(
            x,
            rope,
            permute_mapping=permute_mapping,
            inv_permute_mapping=inv_permute_mapping,
            varlen_handler=varlen_handler,
            local_attn_handler=local_attn_handler,
            modality_dispatcher=modality_dispatcher,
        )
        x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

        x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
        x_video = self.final_norm_video(x_video)
        x_video = self.final_linear_video(x_video)

        x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
        x_audio = self.final_norm_audio(x_audio)
        x_audio = self.final_linear_audio(x_audio)

        x_out = torch.zeros(
            x.shape[0],
            max(self.config.video_in_channels, self.config.audio_in_channels),
            device=x.device,
            dtype=x.dtype,
        )
        x_out[video_mask, : self.config.video_in_channels] = x_video
        x_out[audio_mask, : self.config.audio_in_channels] = x_audio

        return x_out


# ---------------------------------------------------------------------------
# Model config dataclass for building DiTModel
# ---------------------------------------------------------------------------
@dataclass
class MagiHumanDiTConfig:
    """Configuration for building DiTModel, matching the original ModelConfig fields."""

    num_layers: int = 40
    hidden_size: int = 5120
    head_dim: int = 128
    num_query_groups: int = 8
    video_in_channels: int = 48 * 4
    audio_in_channels: int = 64
    text_in_channels: int = 3584
    checkpoint_qk_layernorm_rope: bool = False
    params_dtype: torch.dtype = torch.float32
    mm_layers: list = None
    local_attn_layers: list = None
    enable_attn_gating: bool = True
    gelu7_layers: list = None
    post_norm_layers: list = None

    def __post_init__(self):
        if self.mm_layers is None:
            self.mm_layers = [0, 1, 2, 3, 36, 37, 38, 39]
        if self.local_attn_layers is None:
            self.local_attn_layers = []
        if self.gelu7_layers is None:
            self.gelu7_layers = [0, 1, 2, 3]
        if self.post_norm_layers is None:
            self.post_norm_layers = []
        # Computed fields
        self.num_heads_q = self.hidden_size // self.head_dim
        self.num_heads_kv = self.num_query_groups
