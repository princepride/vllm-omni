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
    """Extended processing info for Omni BAGEL with img2img support.

    For img2img, we use TWO separate modalities:
    - img2img_vae: VAE latent tokens
    - img2img_vit: ViT image tokens

    Each img2img input image becomes 2 virtual input items.
    """

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Support image (for img2text) and split img2img modalities
        # img2img_vae and img2img_vit are separate modalities for the same input
        return {"image": None, "img2img": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        # Max patches for ViT
        max_num_patches = hf_config.vit_max_num_patch_per_side**2

        # For img2img, VAE and ViT are separate modalities
        # VAE tokens: (max_latent_size / latent_patch_size)^2
        latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
        max_latent_size = getattr(hf_config, "max_latent_size", 64)
        max_vae_tokens = (max_latent_size // latent_patch_size) ** 2

        return {
            "image": max_num_patches,
            "img2img_vae": max_vae_tokens,  # VAE tokens only
            "img2img_vit": max_num_patches,  # ViT tokens only
        }


class OmniBagelDataParser(MultiModalDataParser):
    """Custom data parser to support img2img_vae and img2img_vit modalities."""

    def _get_subparsers(self):
        subparsers = super()._get_subparsers()
        # Both img2img_vae and img2img_vit use image parsing logic
        subparsers["img2img_vae"] = self._parse_image_data
        subparsers["img2img_vit"] = self._parse_image_data
        return subparsers


class OmniBagelDummyInputsBuilder(BagelDummyInputsBuilder):
    """Extended dummy inputs builder for Omni BAGEL with img2img support."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Generate dummy text with placeholders for all modalities.

        For img2img, we have TWO placeholders per image:
        - One for VAE (img2img_vae)
        - One for ViT (img2img_vit)
        """
        num_images = mm_counts.get("image", 0)
        num_img2img_vae = mm_counts.get("img2img_vae", 0)
        num_img2img_vit = mm_counts.get("img2img_vit", 0)
        # Each placeholder type gets its own <|image_pad|>
        return "<|image_pad|>" * (num_images + num_img2img_vae + num_img2img_vit)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Generate dummy multimodal data for profiling.

        For img2img, we create dummy data for BOTH img2img_vae and img2img_vit modalities.
        These share the same underlying image data.
        """

        # Get standard image inputs from parent
        result = super().get_dummy_mm_data(seq_len, mm_counts, mm_options)

        # Add img2img dummy inputs for both VAE and ViT modalities
        # They share the same images, so count is the same
        num_img2img_vae = mm_counts.get("img2img_vae", 0)
        num_img2img_vit = mm_counts.get("img2img_vit", 0)

        # Use the max of both (they should be equal in practice)
        num_img2img = max(num_img2img_vae, num_img2img_vit)

        if num_img2img > 0:
            hf_config = self.info.get_hf_config()
            vit_config = hf_config.vit_config
            image_size = vit_config.image_size

            img2img_overrides = mm_options.get("img2img_vae") if mm_options else None

            dummy_images = self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_img2img,
                overrides=img2img_overrides,
            )

            # Both modalities share the same underlying images
            result["img2img_vae"] = dummy_images
            result["img2img_vit"] = dummy_images

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

        Override to handle img2img_vae and img2img_vit modalities.
        Both share the same underlying image data.
        """
        processor_data = dict[str, object]()
        passthrough_data = dict[str, object]()

        for modality, items in mm_items.items():
            if modality in ("img2img_vae", "img2img_vit"):
                # Keep img2img data separate - extract and use "img2img" key
                # Both modalities share the same images, only process once
                if "img2img" not in processor_data:
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
        Custom implementation to handle img2img modalities.

        Processes img2img data and creates pixel_values for both VAE and ViT modalities.
        - ViT: Uses standard HF processor (Siglip)
        - VAE: Uses custom resize (1024x1024) and normalization ([-1, 1])
        """
        import sys

        import numpy as np
        import torch

        # Separate img2img data from other data
        img2img_data = mm_data.get("img2img")
        other_mm_data = {k: v for k, v in mm_data.items() if k != "img2img"}

        # Call upstream processor for text and standard images
        features = super()._call_hf_processor(prompt, other_mm_data, mm_kwargs, tok_kwargs)

        if img2img_data is not None:
            # --- VAE Processing (Custom) ---
            # VAE expects: Resize to 1024x1024 -> ToTensor -> Normalize [-1, 1]
            # Assuming img2img_data is a PIL Image or list of them
            images = img2img_data if isinstance(img2img_data, list) else [img2img_data]
            vae_pixel_values_list = []

            for img in images:
                # Dynamic Resize: Long edge 1024, preserve aspect ratio, align to 16
                MAX_SIZE = 1024
                ALIGNMENT = 16

                w, h = img.size
                scale = MAX_SIZE / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                # Align to multiples of 16 (VAE stride 8 * patch size 2)
                new_w = (new_w // ALIGNMENT) * ALIGNMENT
                new_h = (new_h // ALIGNMENT) * ALIGNMENT

                # Ensure at least one block
                new_w = max(new_w, ALIGNMENT)
                new_h = max(new_h, ALIGNMENT)

                img_resized = img.resize((new_w, new_h))

                # Convert to numpy/tensor and normalize to [-1, 1]
                # PIL gives [0, 255]
                img_arr = np.array(img_resized).astype(np.float32) / 255.0
                img_arr = (img_arr - 0.5) * 2.0  # [0, 1] -> [-1, 1]
                # HWC -> CHW
                img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
                vae_pixel_values_list.append(img_tensor)

            vae_pixel_values = torch.stack(vae_pixel_values_list)
            features["pixel_values_img2img"] = vae_pixel_values

            sys.stderr.write(f"[OmniBagel] Added pixel_values_img2img (VAE) shape: {vae_pixel_values.shape}\n")

        return features

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        """Replace image placeholders with the correct number of tokens.

        For img2img, we use TWO separate modalities:
        - img2img_vae: VAE latent tokens
        - img2img_vit: ViT image tokens

        Each modality has its own PromptReplacement.
        """
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        if image_token_id is None:
            raise ValueError("Image token '<|image_pad|>' not found in tokenizer vocabulary")

        # For img2text: only ViT tokens
        def get_replacement_img2text(item_idx: int):
            num_tokens = hf_config.vit_max_num_patch_per_side**2
            return [image_token_id] * num_tokens

        # For img2img VAE: only VAE tokens
        latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
        max_latent_size = getattr(hf_config, "max_latent_size", 64)
        num_vae_tokens = (max_latent_size // latent_patch_size) ** 2

        def get_replacement_img2img_vae(item_idx: int):
            return [image_token_id] * num_vae_tokens

        # For img2img ViT: only ViT tokens
        num_vit_tokens = hf_config.vit_max_num_patch_per_side**2

        def get_replacement_img2img_vit(item_idx: int):
            return [image_token_id] * num_vit_tokens

        replacements = [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_img2text,
            ),
        ]

        # Check if we have img2img inputs (either modality)
        if "img2img_vae" in mm_items or "img2img_vit" in mm_items or "pixel_values_img2img" in hf_processor_mm_kwargs:
            # Add separate replacements for VAE and ViT
            replacements.append(
                PromptReplacement(
                    modality="img2img_vae",
                    target=[image_token_id],
                    replacement=get_replacement_img2img_vae,
                )
            )
            replacements.append(
                PromptReplacement(
                    modality="img2img_vit",
                    target=[image_token_id],
                    replacement=get_replacement_img2img_vit,
                )
            )

        return replacements

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Define field configs for image and img2img modalities.

        For img2img, we create TWO field entries from the same pixel_values_img2img:
        - pixel_values_img2img -> img2img_vae
        - pixel_values_img2img_vit -> img2img_vit (we duplicate the data)
        """
        config = {
            "pixel_values": MultiModalFieldConfig.batched("image"),
        }

        # Add img2img fields if present
        # Both VAE and ViT modalities share the same underlying pixel data
        if "pixel_values_img2img" in hf_inputs:
            # Use pixel_values_img2img for VAE modality
            config["pixel_values_img2img"] = MultiModalFieldConfig.batched("img2img_vae")
            # Duplicate for ViT modality - both will process the same images
            config["pixel_values_img2img_vit"] = MultiModalFieldConfig.batched("img2img_vit")

        return config
