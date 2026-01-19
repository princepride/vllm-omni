# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model-specific extractors for TeaCache.

This module provides a registry of extractor functions that know how to extract
modulated inputs from different transformer architectures. Adding support for
a new model requires only adding a new extractor function to the registry.

With Option B enhancement, extractors now return a CacheContext object containing
all model-specific information needed for generic caching, including preprocessing,
transformer execution, and postprocessing logic.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CacheContext:
    """
    Context object containing all model-specific information for caching.

    This allows the TeaCacheHook to remain completely generic - all model-specific
    logic is encapsulated in the extractor that returns this context.

    Attributes:
        modulated_input: Tensor used for cache decision (similarity comparison).
            Must be a torch.Tensor extracted from the first transformer block,
            typically after applying normalization and modulation.

        hidden_states: Current hidden states (will be modified by caching).
            Must be a torch.Tensor representing the main image/latent states
            after preprocessing but before transformer blocks.

        encoder_hidden_states: Optional encoder states (for dual-stream models).
            Set to None for single-stream models (e.g., Flux).
            For dual-stream models (e.g., Qwen), contains text encoder outputs.

        temb: Timestep embedding tensor.
            Must be a torch.Tensor containing the timestep conditioning.

        run_transformer_blocks: Callable that executes model-specific transformer blocks.
            Signature: () -> tuple[torch.Tensor, ...]

            Returns:
                tuple containing:
                - [0]: processed hidden_states (required)
                - [1]: processed encoder_hidden_states (optional, only for dual-stream)

            Example for single-stream:
                def run_blocks():
                    h = hidden_states
                    for block in module.transformer_blocks:
                        h = block(h, temb=temb)
                    return (h,)

            Example for dual-stream:
                def run_blocks():
                    h, e = hidden_states, encoder_hidden_states
                    for block in module.transformer_blocks:
                        e, h = block(h, e, temb=temb)
                    return (h, e)

        postprocess: Callable that does model-specific output postprocessing.
            Signature: (torch.Tensor) -> Union[torch.Tensor, Transformer2DModelOutput, tuple]

            Takes the processed hidden_states and applies final transformations
            (normalization, projection) to produce the model output.

            Example:
                def postprocess(h):
                    h = module.norm_out(h, temb)
                    output = module.proj_out(h)
                    return Transformer2DModelOutput(sample=output)

        extra_states: Optional dict for additional model-specific state.
            Use this for models that need to pass additional context beyond
            the standard fields.
    """

    modulated_input: torch.Tensor
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor | None
    temb: torch.Tensor
    run_transformer_blocks: Callable[[], tuple[torch.Tensor, ...]]
    postprocess: Callable[[torch.Tensor], Any]
    extra_states: dict[str, Any] | None = None

    def validate(self) -> None:
        """
        Validate that the CacheContext contains valid data.

        Raises:
            TypeError: If fields have wrong types
            ValueError: If tensors have invalid properties
            RuntimeError: If callables fail basic invocation tests

        This method should be called after creating a CacheContext to catch
        common developer errors early with clear error messages.
        """
        # Validate tensor fields
        if not isinstance(self.modulated_input, torch.Tensor):
            raise TypeError(f"modulated_input must be torch.Tensor, got {type(self.modulated_input)}")

        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be torch.Tensor, got {type(self.hidden_states)}")

        if self.encoder_hidden_states is not None and not isinstance(self.encoder_hidden_states, torch.Tensor):
            raise TypeError(
                f"encoder_hidden_states must be torch.Tensor or None, got {type(self.encoder_hidden_states)}"
            )

        if not isinstance(self.temb, torch.Tensor):
            raise TypeError(f"temb must be torch.Tensor, got {type(self.temb)}")

        # Validate callables
        if not callable(self.run_transformer_blocks):
            raise TypeError(f"run_transformer_blocks must be callable, got {type(self.run_transformer_blocks)}")

        if not callable(self.postprocess):
            raise TypeError(f"postprocess must be callable, got {type(self.postprocess)}")

        # Validate tensor shapes are compatible
        if self.modulated_input.shape[0] != self.hidden_states.shape[0]:
            raise ValueError(
                f"Batch size mismatch: modulated_input has batch size "
                f"{self.modulated_input.shape[0]}, but hidden_states has "
                f"{self.hidden_states.shape[0]}"
            )

        # Validate devices match
        if self.modulated_input.device != self.hidden_states.device:
            raise ValueError(
                f"Device mismatch: modulated_input on {self.modulated_input.device}, "
                f"hidden_states on {self.hidden_states.device}"
            )


def extract_qwen_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor | float | int,
    img_shapes: torch.Tensor,
    txt_seq_lens: torch.Tensor,
    guidance: torch.Tensor | None = None,
    additional_t_cond: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for QwenImageTransformer2DModel.

    This is the ONLY Qwen-specific code needed for TeaCache support.
    It encapsulates preprocessing, modulated input extraction, transformer execution,
    and postprocessing logic.

    Args:
        module: QwenImageTransformer2DModel instance
        hidden_states: Input hidden states tensor
        encoder_hidden_states: Text encoder outputs
        encoder_hidden_states_mask: Mask for text encoder
        timestep: Current diffusion timestep
        img_shapes: Image shapes for position embedding
        txt_seq_lens: Text sequence lengths
        guidance: Optional guidance scale for CFG
        additional_t_cond: Optional additional timestep conditioning
        attention_kwargs: Additional attention arguments
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    # ============================================================================
    # PREPROCESSING (Qwen-specific)
    # ============================================================================
    hidden_states = module.img_in(hidden_states)
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
    encoder_hidden_states = module.txt_norm(encoder_hidden_states)
    encoder_hidden_states = module.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        module.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else module.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    image_rotary_emb = module.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    block = module.transformer_blocks[0]
    img_mod_params = block.img_mod(temb)
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)
    img_modulated, _ = block.img_norm1(hidden_states, img_mod1)

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Qwen-specific)
    # ============================================================================
    def run_transformer_blocks():
        """Execute all Qwen transformer blocks."""
        h = hidden_states
        e = encoder_hidden_states
        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )
        return (h, e)

    # ============================================================================
    # DEFINE POSTPROCESSING (Qwen-specific)
    # ============================================================================
    return_dict = kwargs.get("return_dict", True)

    def postprocess(h):
        """Apply Qwen-specific output postprocessing."""
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_bagel_context(
    module: nn.Module,
    x_t: torch.Tensor,
    timestep: torch.Tensor | float | int,
    packed_vae_token_indexes: torch.LongTensor,
    packed_vae_position_ids: torch.LongTensor,
    packed_text_ids: torch.LongTensor,
    packed_text_indexes: torch.LongTensor,
    packed_indexes: torch.LongTensor,
    packed_position_ids: torch.LongTensor,
    packed_seqlens: torch.IntTensor,
    key_values_lens: torch.IntTensor,
    past_key_values: Any,
    packed_key_value_indexes: torch.LongTensor,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Bagel model.

    Args:
        module: Bagel instance
        x_t: Latent image input
        timestep: Current timestep
        packed_vae_token_indexes: Indexes for VAE tokens in packed sequence
        packed_vae_position_ids: Position IDs for VAE tokens
        packed_text_ids: Text token IDs
        packed_text_indexes: Indexes for text tokens in packed sequence
        packed_indexes: Global indexes
        packed_position_ids: Global position IDs
        packed_seqlens: Sequence lengths
        key_values_lens: KV cache lengths
        past_key_values: KV cache
        packed_key_value_indexes: KV cache indexes
        **kwargs: Additional keyword arguments

    Returns:
        CacheContext with all information needed for generic caching
    """
    # 1. Embed text
    packed_text_embedding = module.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), module.hidden_size))
    packed_sequence[packed_text_indexes] = packed_text_embedding

    # 2. Embed timestep
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=x_t.device)
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)

    # 3. Embed image (x_t)
    packed_pos_embed = module.latent_pos_embed(packed_vae_position_ids)
    packed_timestep_embeds = module.time_embedder(timestep)

    x_t_emb = module.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
    if x_t_emb.dtype != packed_sequence.dtype:
        x_t_emb = x_t_emb.to(packed_sequence.dtype)

    packed_sequence[packed_vae_token_indexes] = x_t_emb

    # Use the full packed sequence as modulated input to match hidden_states size
    modulated_input = packed_sequence

    def run_transformer_blocks():
        extra_inputs = {}
        if module.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = module.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        return (output.packed_query_sequence,)

    def postprocess(h):
        v_t = module.llm2vae(h)
        v_t = v_t[packed_vae_token_indexes]
        return v_t

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=packed_sequence,  # Use full packed sequence
        encoder_hidden_states=None,
        temb=packed_timestep_embeds,  # Approximate
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


# Registry for model-specific extractors
# Key: Transformer class name
# Value: extractor function with signature (module, *args, **kwargs) -> CacheContext
#
# Note: Use the transformer class name as specified in pipelines as TeaCache hooks operate
# on the transformer module and multiple pipelines can share the same transformer.
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "QwenImageTransformer2DModel": extract_qwen_context,
    "Bagel": extract_bagel_context,
    # Future models:
    # "FluxTransformer2DModel": extract_flux_context,
    # "CogVideoXTransformer3DModel": extract_cogvideox_context,
}


def register_extractor(transformer_cls_name: str, extractor_fn: Callable) -> None:
    """
    Register a new extractor function for a model type.

    This allows extending TeaCache support to new models without modifying
    the core TeaCache code.

    Args:
        transformer_cls_name: Transformer model type identifier (class name or type string)
        extractor_fn: Function with signature (module, *args, **kwargs) -> CacheContext

    Example:
        >>> def extract_flux_context(module, hidden_states, timestep, guidance=None, **kwargs):
        ...     # Preprocessing
        ...     temb = module.time_text_embed(timestep, guidance)
        ...     # Extract modulated input
        ...     modulated = module.transformer_blocks[0].norm1(hidden_states, emb=temb)
        ...     # Define execution
        ...     def run_blocks():
        ...         h = hidden_states
        ...         for block in module.transformer_blocks:
        ...             h = block(h, temb=temb)
        ...         return (h,)
        ...     # Define postprocessing
        ...     def postprocess(h):
        ...         return module.proj_out(module.norm_out(h, temb))
        ...     # Return context
        ...     return CacheContext(modulated, hidden_states, None, temb, run_blocks, postprocess)
        >>> register_extractor("FluxTransformer2DModel", extract_flux_context)
    """
    EXTRACTOR_REGISTRY[transformer_cls_name] = extractor_fn


def get_extractor(transformer_cls_name: str) -> Callable:
    """
    Get extractor function for given transformer class.

    This function looks up the extractor based on the exact transformer_cls_name string,
    which should match the transformer type in the pipeline (i.e., pipeline.transformer.__class__.__name__).

    Args:
        transformer_cls_name: Transformer class name (e.g., "QwenImageTransformer2DModel")
                                Must exactly match a key in EXTRACTOR_REGISTRY.

    Returns:
        Extractor function with signature (module, *args, **kwargs) -> CacheContext

    Raises:
        ValueError: If model type not found in registry

    Example:
        >>> # Get extractor for QwenImageTransformer2DModel
        >>> extractor = get_extractor("QwenImageTransformer2DModel")
        >>> ctx = extractor(transformer, hidden_states, encoder_hidden_states, timestep, ...)
    """
    # Direct lookup - no substring matching
    if transformer_cls_name in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[transformer_cls_name]

    # No match found
    available_types = list(EXTRACTOR_REGISTRY.keys())
    raise ValueError(
        f"Unknown model type: '{transformer_cls_name}'. "
        f"Available types: {available_types}\n"
        f"To add support for a new model, use register_extractor() or add to EXTRACTOR_REGISTRY."
    )
