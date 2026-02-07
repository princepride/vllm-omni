"""
OmniBagelForConditionalGeneration - Extended BAGEL model for img2img support.

This module extends the upstream vLLM BagelForConditionalGeneration to support
img2img tasks by adding VAE encoder for latent computation.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.bagel import BagelForConditionalGeneration
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.diffusion.models.bagel.autoencoder import (
    AutoEncoderParams,
    DiagonalGaussian,
    Encoder,
)
from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    PositionEmbedding,
    TimestepEmbedder,
)
from vllm_omni.model_executor.models.bagel.processor import (
    OmniBagelDummyInputsBuilder,
    OmniBagelMultiModalProcessor,
    OmniBagelProcessingInfo,
)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


def default_ae_params() -> AutoEncoderParams:
    """Default VAE parameters matching BAGEL checkpoint."""
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


class VAEEncoder(nn.Module):
    """
    VAE Encoder-only wrapper for img2img latent computation.

    Only contains the encoder portion of the AutoEncoder to avoid loading
    decoder weights which are not needed for img2img embedding computation.
    """

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
        self.reg = DiagonalGaussian()
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z


@MULTIMODAL_REGISTRY.register_processor(
    OmniBagelMultiModalProcessor,
    info=OmniBagelProcessingInfo,
    dummy_inputs=OmniBagelDummyInputsBuilder,
)
class OmniBagelForConditionalGeneration(BagelForConditionalGeneration):
    """
    Omni version of BagelForConditionalGeneration with img2img support.

    Extends the upstream vLLM BAGEL to support:
    - VAE encoder for latent computation in img2img tasks
    - Combined VAE + ViT embeddings for img2img modality
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config

        # Initialize VAE-related modules for img2img support
        # VAE weights are in a separate ae.safetensors file which must be loaded
        # Enable VAE by default for img2img support
        enable_vae = getattr(config, "load_vae_weights", True) and getattr(config, "visual_gen", True)

        if enable_vae:
            # VAE configuration
            self.latent_patch_size = getattr(config, "latent_patch_size", 2)
            self.latent_downsample = 8 * self.latent_patch_size  # vae_downsample * patch_size
            # NOTE: Config says max_latent_size=32, but checkpoint latent_pos_embed has 4096=64^2 positions
            # Override to 64 to match the actual checkpoint weights
            config_max_latent_size = getattr(config, "max_latent_size", 32)
            self.max_latent_size = 64  # Force 64 to match checkpoint
            if config_max_latent_size != 64:
                logger.warning(f"Overriding max_latent_size from {config_max_latent_size} to 64 to match checkpoint")
            self.latent_channel = 16  # z_channels from VAE config

            hidden_size = config.llm_config.hidden_size
            patch_latent_dim = self.latent_patch_size**2 * self.latent_channel

            # VAE encoder only (decoder is in diffusion stage, not needed here)
            self.vae = VAEEncoder(default_ae_params())

            # Projection from VAE latent to LLM hidden size
            self.vae2llm = nn.Linear(patch_latent_dim, hidden_size)

            # Position embedding for VAE latent tokens
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, hidden_size)

            # Timestep embedder (for img2img we use t=0)
            self.time_embedder = TimestepEmbedder(hidden_size)

            self._img2img_enabled = True
            logger.info("OmniBagelForConditionalGeneration: img2img VAE encoder initialized")
        else:
            self._img2img_enabled = False
            if getattr(config, "visual_gen", True):
                logger.info(
                    "OmniBagelForConditionalGeneration: VAE disabled (set load_vae_weights=True in config to enable)"
                )
            else:
                logger.info("OmniBagelForConditionalGeneration: img2img disabled (visual_gen=False)")

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        """Parse multimodal inputs and separate by modality.

        Supports:
        - img2text: standard image understanding (ViT only)
        - img2img_vae: VAE embeddings for img2img
        - img2img_vit: ViT embeddings for img2img
        """
        mm_input_by_modality = {}

        # Parse img2text (standard image understanding)
        if any(k in kwargs for k in ("pixel_values", "image_embeds")):
            mm_input_by_modality["img2text"] = self._parse_and_validate_image_input(**kwargs)

        # Parse img2img_vae - maps pixel_values_img2img -> pixel_values
        if "pixel_values_img2img" in kwargs:
            vae_kwargs = kwargs.copy()
            vae_kwargs["pixel_values"] = kwargs["pixel_values_img2img"]
            mm_input_by_modality["img2img_vae"] = self._parse_and_validate_image_input(**vae_kwargs)

        # Parse img2img_vit - maps pixel_values_img2img_vit -> pixel_values
        if "pixel_values_img2img_vit" in kwargs:
            vit_kwargs = kwargs.copy()
            vit_kwargs["pixel_values"] = kwargs["pixel_values_img2img_vit"]
            mm_input_by_modality["img2img_vit"] = self._parse_and_validate_image_input(**vit_kwargs)

        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        """
        Compute multimodal embeddings for all modalities.

        For img2text: Uses ViT + connector (standard image understanding)
        For img2img_vae: Uses VAE encoder (latent embeddings)
        For img2img_vit: Uses ViT + connector (same as img2text)
        """
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "img2text":
                image_embeddings = self._process_img2text_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            elif modality == "img2img_vae":
                vae_embeddings = self._process_img2img_vae_input(multimodal_input)
                multimodal_embeddings += tuple(vae_embeddings)
            elif modality == "img2img_vit":
                vit_embeddings = self._process_img2img_vit_input(multimodal_input)
                multimodal_embeddings += tuple(vit_embeddings)

        return multimodal_embeddings

    def _process_img2text_input(self, multimodal_input):
        """Process image for understanding task (ViT only)."""
        return self._process_image_input(multimodal_input)

    def _process_img2img_vae_input(self, multimodal_input) -> tuple[torch.Tensor, ...]:
        """
        Process image for img2img task - VAE embeddings only.

        Returns VAE latent embeddings for each image.
        Each image -> one tensor of shape (num_vae_tokens, hidden_size)
        """
        if not self._img2img_enabled:
            logger.warning("img2img not enabled, returning empty")
            return ()

        pixel_values = multimodal_input["pixel_values"]
        # Ensure pixel_values matches VAE dtype (e.g. bfloat16)
        pixel_values = pixel_values.to(dtype=next(self.vae.parameters()).dtype)
        logger.info(f"[Stage0 img2img_vae] Input pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")

        # Handle potential extra batch dimension
        if pixel_values.ndim == 5:
            batch_size, num_images, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_images, channels, height, width)
        else:
            batch_size = pixel_values.shape[0]

        # ========== VAE Latent Embeddings ==========
        # Encode image to latent space
        with torch.no_grad():
            # VAE expects input in range [-1, 1], pixel_values should already be normalized
            latent = self.vae.encode(pixel_values)  # (B, C, H, W)
            logger.info(f"[Stage0 img2img_vae] VAE latent shape: {latent.shape}")

        # Patchify latent
        p = self.latent_patch_size
        B, C, H, W = latent.shape
        h, w = H // p, W // p

        # Rearrange: (B, C, H, W) -> (B, h*w, p*p*C)
        latent_patches = rearrange(latent, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

        # Project to LLM hidden size
        vae_embeds = self.vae2llm(latent_patches)  # (B, num_patches, hidden_size)
        logger.info(f"[Stage0 img2img_vae] VAE projected embeds shape: {vae_embeds.shape}")

        # Add position embeddings (2D grid positions)
        num_patches = h * w
        # Create position IDs for 2D grid
        h_coords = torch.arange(h, device=vae_embeds.device)
        w_coords = torch.arange(w, device=vae_embeds.device)
        position_ids = (h_coords[:, None] * self.max_latent_size + w_coords).flatten()
        position_ids = position_ids.unsqueeze(0).expand(B, -1).reshape(-1)

        pos_embeds = self.latent_pos_embed(position_ids)
        pos_embeds = pos_embeds.reshape(B, num_patches, -1)

        # Add timestep embedding (t=0 for img2img condition)
        # Cast to model dtype to avoid Float vs BFloat16 mismatch
        timestep = torch.zeros(B, device=vae_embeds.device, dtype=vae_embeds.dtype)
        time_embeds = self.time_embedder(timestep)  # (B, hidden_size)
        time_embeds = time_embeds.unsqueeze(1)  # (B, 1, hidden_size)

        vae_embeds = vae_embeds + pos_embeds + time_embeds

        # Return VAE embeddings as separate items per image
        vae_embeddings = []
        for i in range(B):
            vae_embed = vae_embeds[i]  # (num_vae_tokens, hidden_size)
            vae_embeddings.append(vae_embed)
            logger.info(f"[Stage0 img2img_vae] Image {i} final VAE embedding shape: {vae_embed.shape}")

        logger.info(f"[Stage0 img2img_vae] Total VAE embeddings count: {len(vae_embeddings)}")
        return tuple(vae_embeddings)

    def _process_img2img_vit_input(self, multimodal_input) -> tuple[torch.Tensor, ...]:
        """
        Process image for img2img task - ViT embeddings only.

        Returns ViT embeddings for each image (same as img2text processing).
        Each image -> one tensor of shape (num_vit_tokens, hidden_size)
        """
        # ViT processing is the same as img2text
        vit_embeddings = self._process_image_input(multimodal_input)

        logger.info(f"[Stage0 img2img_vit] ViT embeds count: {len(vit_embeddings)}")
        if len(vit_embeddings) > 0:
            logger.info(f"[Stage0 img2img_vit] Image 0 ViT embedding shape: {vit_embeddings[0].shape}")

        return vit_embeddings

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for OmniBagelForConditionalGeneration.

        When VAE is disabled, we simply delegate to the parent class
        to ensure all standard mappings (including ViT) are handled correctly.
        When VAE is enabled, we extend the loading logic to handle VAE weights.
        """
        if not self._img2img_enabled:
            return super().load_weights(weights)

        # Logic for when VAE is enabled
        # We need to filter weights but KEEP VAE weights if present.
        # And we must use the correct mapper for the base model.

        generation_keywords_to_skip = [
            "moe_gen",
            "llm2vae",
            "decoder.",  # Skip VAE decoder weights (only need encoder for img2img)
        ]

        def _map_vae_weight_name(name: str) -> str:
            """Map VAE weight names from ae.safetensors format to model format."""
            # ae.safetensors stores weights as 'encoder.*' and 'reg.*' (DiagonalGaussian)
            # We need to map them to 'vae.encoder.*' and 'vae.reg.*'
            if name.startswith("encoder."):
                return "vae." + name
            if name.startswith("reg."):
                return "vae." + name
            # Also handle vae2llm, latent_pos_embed, time_embedder from main checkpoint
            return name

        filtered_weights = []
        for name, tensor in weights:
            if any(skip in name for skip in generation_keywords_to_skip):
                continue

            # Map VAE weight names
            mapped_name = _map_vae_weight_name(name)

            # Handle patch embedding reshape for SigLIP (same as upstream)
            if "patch_embedding.weight" in mapped_name and tensor.ndim == 2:
                out_channels = tensor.shape[0]
                in_features = tensor.shape[1]
                patch_size = self.config.vit_config.patch_size
                in_channels = self.config.vit_config.num_channels
                if in_features == in_channels * patch_size * patch_size:
                    tensor = tensor.reshape(out_channels, patch_size, patch_size, in_channels)
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()

            filtered_weights.append((mapped_name, tensor))

        # Use the parent's mapper for standard components
        # VAE-related weights may not be in the main checkpoint (they're in ae.safetensors)
        # Mark them as optional using ignore_unexpected_prefixes
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["vit_pos_embed.pos_embed"],
            ignore_unexpected_prefixes=["vae.", "latent_pos_embed.", "time_embedder.", "vae2llm."],
        )
        return loader.load_weights(filtered_weights, mapper=self.hf_to_vllm_mapper)
