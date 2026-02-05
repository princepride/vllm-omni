"""
Custom multimodal processor for OmniBagelForConditionalGeneration with img2img support.

This processor extends the vLLM BagelMultiModalProcessor to support both:
- img2text (image understanding): modality "image" with pixel_values
- img2img (image-to-image generation): modality "img2img" with pixel_values_img2img
"""

from collections.abc import Mapping, Sequence
from typing import Any

from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import PromptReplacement

# Import upstream components
try:
    from vllm.model_executor.models.bagel import (
        BagelDummyInputsBuilder,
        BagelMultiModalProcessor,
        BagelProcessingInfo,
    )
except ImportError:
    # Fallback for older vLLM versions
    from vllm.transformers_utils.processors.bagel import BagelProcessor as BagelMultiModalProcessor

    BagelProcessingInfo = None
    BagelDummyInputsBuilder = None


class OmniBagelProcessingInfo(BagelProcessingInfo):
    """Extended processing info for Omni BAGEL with img2img support."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Support both image (for img2text) and img2img modalities
        return {"image": None, "img2img": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        # Max patches for ViT
        max_num_patches = hf_config.vit_max_num_patch_per_side**2

        # For img2img, we need VAE tokens + ViT tokens
        # VAE tokens: (max_latent_size / latent_patch_size)^2
        latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
        max_latent_size = getattr(hf_config, "max_latent_size", 64)  # 64 matches checkpoint
        max_vae_tokens = (max_latent_size // latent_patch_size) ** 2

        return {
            "image": max_num_patches,
            "img2img": max_vae_tokens + max_num_patches,  # VAE + ViT tokens
        }


class OmniBagelDataParser(MultiModalDataParser):
    """Custom data parser to support img2img modality."""

    def _get_subparsers(self):
        subparsers = super()._get_subparsers()
        # img2img uses same parsing logic as image (returns Image object)
        subparsers["img2img"] = self._parse_image_data
        return subparsers


class OmniBagelDummyInputsBuilder(BagelDummyInputsBuilder):
    """Extended dummy inputs builder for Omni BAGEL with img2img support."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Generate dummy text with placeholders for all modalities."""
        num_images = mm_counts.get("image", 0)
        num_img2img = mm_counts.get("img2img", 0)
        # Use a simple placeholder for each image/img2img
        return "<|image_pad|>" * (num_images + num_img2img)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Generate dummy multimodal data for profiling."""

        # Get standard image inputs from parent
        result = super().get_dummy_mm_data(seq_len, mm_counts, mm_options)

        # Add img2img dummy inputs if requested
        num_img2img = mm_counts.get("img2img", 0)
        if num_img2img > 0:
            hf_config = self.info.get_hf_config()
            vit_config = hf_config.vit_config
            image_size = vit_config.image_size

            img2img_overrides = mm_options.get("img2img") if mm_options else None

            result["img2img"] = self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_img2img,
                overrides=img2img_overrides,
            )

        return result


class OmniBagelMultiModalProcessor(BagelMultiModalProcessor):
    """
    Extended multimodal processor for BAGEL with img2img support.

    Handles both:
    - "image" modality for img2text (understanding)
    - "img2img" modality for image-to-image generation
    """

    def __init__(self, info: OmniBagelProcessingInfo, dummy_inputs_builder: OmniBagelDummyInputsBuilder, **kwargs):
        super().__init__(info, dummy_inputs_builder, **kwargs)
        # Override data parser with custom one supporting img2img
        self.data_parser = OmniBagelDataParser()

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        """
        Extract processor data from multimodal items.

        Override to keep img2img data separate from regular images.
        When img2img items are parsed as ImageItems, their get_processor_data()
        returns {"images": ...}, which would merge with regular images.
        We need to preserve img2img under its own key.
        """
        processor_data = dict[str, object]()
        passthrough_data = dict[str, object]()

        for modality, items in mm_items.items():
            if modality == "img2img":
                # Keep img2img separate - extract the data and use "img2img" key
                item_data = items.get_processor_data()
                # ImageItems.get_processor_data() returns {"images": ...}
                if "images" in item_data:
                    processor_data["img2img"] = item_data["images"]
                passthrough_data.update(items.get_passthrough_data())
            else:
                processor_data.update(items.get_processor_data())
                passthrough_data.update(items.get_passthrough_data())

        return processor_data, passthrough_data

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Any:  # Returns BatchFeature
        """
        Custom implementation to handle img2img modality.

        Splits img2img data, processes it as 'images' to get pixel_values,
        then renames to 'pixel_values_img2img'.
        """
        import sys

        # Separate img2img data from other data
        img2img_data = mm_data.get("img2img")
        other_mm_data = {k: v for k, v in mm_data.items() if k != "img2img"}

        # Call upstream processor for text and standard images
        features = super()._call_hf_processor(prompt, other_mm_data, mm_kwargs, tok_kwargs)

        if img2img_data is not None:
            # Process img2img data using the same processor, but mapping to "images"
            hf_processor = self.info.get_hf_processor(**mm_kwargs)

            # We call the processor with "images" argument
            # vLLM's call_hf_processor wrapper handles the call
            img2img_features = self.info.ctx.call_hf_processor(
                hf_processor,
                {"text": prompt, "images": img2img_data},
                {**mm_kwargs, **tok_kwargs},
            )

            sys.stderr.write(f"[OmniBagel] img2img_features keys: {list(img2img_features.keys())}\n")

            # Extract and rename pixel_values
            if "pixel_values" in img2img_features:
                features["pixel_values_img2img"] = img2img_features["pixel_values"]
                sys.stderr.write(
                    f"[OmniBagel] Added pixel_values_img2img with shape: {features['pixel_values_img2img'].shape}\n"
                )
            else:
                sys.stderr.write(
                    f"[OmniBagel] pixel_values MISSING in img2img_features. Keys: {list(img2img_features.keys())}\n"
                )

        return features

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        """Replace image placeholders with the correct number of tokens."""
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        if image_token_id is None:
            raise ValueError("Image token '<|image_pad|>' not found in tokenizer vocabulary")

        # For img2text: only ViT tokens
        def get_replacement_img2text(item_idx: int):
            num_tokens = hf_config.vit_max_num_patch_per_side**2
            return [image_token_id] * num_tokens

        # For img2img: Check if VAE is enabled (enabled by default)
        vae_enabled = getattr(hf_config, "load_vae_weights", True) and getattr(hf_config, "visual_gen", True)

        def get_replacement_img2img(item_idx: int):
            num_vit_tokens = hf_config.vit_max_num_patch_per_side**2
            if vae_enabled:
                # VAE tokens + ViT tokens when VAE is enabled
                latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
                max_latent_size = getattr(hf_config, "max_latent_size", 64)  # 64 matches checkpoint
                num_vae_tokens = (max_latent_size // latent_patch_size) ** 2
                return [image_token_id] * (num_vae_tokens + num_vit_tokens)
            else:
                # Only ViT tokens when VAE is disabled (fallback to img2text behavior)
                return [image_token_id] * num_vit_tokens

        replacements = [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_img2text,
            ),
        ]

        # Check if we have img2img inputs
        if "img2img" in mm_items or "pixel_values_img2img" in hf_processor_mm_kwargs:
            # Add img2img replacement
            # We use a different placeholder token if available, otherwise same
            img2img_token_id = tokenizer.get_vocab().get("<|img2img_pad|>", image_token_id)
            replacements.append(
                PromptReplacement(
                    modality="img2img",
                    target=[img2img_token_id],
                    replacement=get_replacement_img2img,
                )
            )

        return replacements

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Define field configs for both image and img2img modalities."""
        config = {
            "pixel_values": MultiModalFieldConfig.batched("image"),
        }

        # Add img2img field if present
        if "pixel_values_img2img" in hf_inputs:
            config["pixel_values_img2img"] = MultiModalFieldConfig.batched("img2img")

        return config
