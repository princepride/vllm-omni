from collections.abc import Iterable, Mapping, Sequence
from math import isqrt
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.bagel import BagelForConditionalGeneration
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
)
from vllm.transformers_utils.processors.bagel import BagelProcessor

from vllm_omni.diffusion.models.bagel.autoencoder import (
    AutoEncoderParams,
    DiagonalGaussian,
    Encoder,
)
from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    PositionEmbedding,
    TimestepEmbedder,
)
from vllm_omni.diffusion.models.bagel.pipeline_bagel import default_ae_params


class OmniBagelProcessor(BagelProcessor):
    def __call__(self, text=None, images=None, **kwargs):
        is_img2img = kwargs.pop("is_img2img", False)

        if is_img2img and images is not None:
            # Manually handle img2img to separate kwargs

            # Image Processor Args
            image_kwargs = kwargs.copy()
            image_kwargs["do_resize"] = False
            image_kwargs["do_rescale"] = True
            if "return_tensors" not in image_kwargs:
                image_kwargs["return_tensors"] = "pt"

            pixel_values = self.image_processor(images, **image_kwargs)

            # Tokenizer Args (original kwargs without do_resize)
            text_inputs = self.tokenizer(text, **kwargs) if text is not None else None

            # Combine
            if pixel_values is not None and text_inputs is not None:
                combined = dict(text_inputs)
                combined["pixel_values"] = pixel_values["pixel_values"]
                return BatchFeature(combined)
            elif pixel_values is not None:
                return pixel_values
            elif text_inputs is not None:
                return BatchFeature(dict(text_inputs))
            else:
                return BatchFeature({})

        return super().__call__(text, images, **kwargs)


class OmniBagelProcessingInfo(BaseProcessingInfo):
    """Processing info for OmniBagelForConditionalGeneration."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1, "img2img": 1}

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(OmniBagelProcessor, **kwargs)


class OmniBagelDummyInputsBuilder(BaseDummyInputsBuilder[OmniBagelProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        dummy_text = ""
        if "image" in mm_counts:
            dummy_text += "<|image_pad|>" * mm_counts["image"]
        if "img2img" in mm_counts:
            dummy_text += "<|fim_middle|>" * mm_counts["img2img"]
        return dummy_text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        hf_config = self.info.get_hf_config()
        vit_config = hf_config.vit_config

        # Use the configured image size
        image_size = vit_config.image_size
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "img2img": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=mm_counts.get("img2img", 0),
                overrides=image_overrides,
            ),
        }


class Img2ImgProcessorItems(ImageProcessorItems):
    def __init__(self, data):
        super().__init__(data)
        self.modality = "img2img"

    def get_processor_data(self):
        # customized key for img2img to avoid collision with image
        return {"pixel_values_img2img": self.get_all()}


class OmniBagelDataParser(MultiModalDataParser):
    def _parse_img2img_data(self, data: ModalityData) -> ModalityDataItems | None:
        # Reuse image parsing logic but wrap in Img2ImgProcessorItems
        items = self._parse_image_data(data)
        if items is None:
            return None
        # Convert ImageProcessorItems to Img2ImgProcessorItems
        return Img2ImgProcessorItems(items.data)

    def _get_subparsers(self):
        parsers = super()._get_subparsers()
        parsers["img2img"] = self._parse_img2img_data
        return parsers


class OmniBagelMultiModalProcessor(BaseMultiModalProcessor[OmniBagelProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return OmniBagelDataParser(
            expected_hidden_size=self.info.get_hf_config().hidden_size,
        )

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "pixel_values_img2img": MultiModalFieldConfig.batched("img2img"),
        }

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        has_image = "images" in mm_data
        has_img2img = "pixel_values_img2img" in mm_data

        # If we have both, we need to process separately and merge
        if has_image and has_img2img:
            outputs = BatchFeature()

            # 1. Process standard images
            img_data = dict(mm_data)
            # Remove img2img data to avoid confusion, keep "images"
            if "pixel_values_img2img" in img_data:
                del img_data["pixel_values_img2img"]

            # Force is_img2img=False for standard images
            kwargs_img = dict(mm_kwargs)
            kwargs_img["is_img2img"] = False

            out_img = super()._call_hf_processor(prompt, img_data, kwargs_img, tok_kwargs)

            if "pixel_values" in out_img:
                outputs["pixel_values"] = out_img["pixel_values"]
            for k, v in out_img.items():
                if k != "pixel_values":
                    outputs[k] = v

            # 2. Process img2img
            img2img_data = dict(mm_data)
            # Remove standard images to avoid confusion
            if "images" in img2img_data:
                del img2img_data["images"]

            # Rename for processor
            img2img_data["images"] = img2img_data.pop("pixel_values_img2img")

            kwargs_img2img = dict(mm_kwargs)
            kwargs_img2img["is_img2img"] = True

            out_img2img = super()._call_hf_processor(prompt, img2img_data, kwargs_img2img, tok_kwargs)

            if "pixel_values" in out_img2img:
                outputs["pixel_values_img2img"] = out_img2img["pixel_values"]

            # Merge other keys (should be same for text)
            for k, v in out_img2img.items():
                if k not in outputs:
                    outputs[k] = v

            return outputs

        elif has_img2img:
            mm_data = dict(mm_data)
            mm_data["images"] = mm_data.pop("pixel_values_img2img")
            # If standard images were somehow present but not detected by "images" key check (unlikely), remove them?
            # No, if has_image is false, "images" is not in mm_data.

            mm_kwargs = dict(mm_kwargs)
            mm_kwargs["is_img2img"] = True

            outputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

            if "pixel_values" in outputs:
                outputs["pixel_values_img2img"] = outputs.pop("pixel_values")

            return outputs

        # Standard image case or text only case
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        """Replace image placeholders with the correct number of tokens."""
        hf_config = self.info.get_hf_config()

        # Get the tokenizer to look up the image token ID
        tokenizer = self.info.get_tokenizer()
        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        img2img_token_id = tokenizer.get_vocab().get("<|fim_middle|>")
        img2img_vision_start = tokenizer.get_vocab().get("<|vision_start|>")
        img2img_vision_end = tokenizer.get_vocab().get("<|vision_end|>")

        def get_replacement_img2text_bagel(item_idx: int):
            # For BAGEL, calculate number of tokens based on max patch size
            num_tokens = hf_config.vit_max_num_patch_per_side**2
            # Use the image token ID from tokenizer
            return [image_token_id] * num_tokens

        def get_replacement_img2img_bagel(item_idx: int):
            images = mm_items.get_items("img2img", (ImageEmbeddingItems, Img2ImgProcessorItems))
            if isinstance(images, ImageEmbeddingItems):
                num_vae_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                latent_downsample = hf_config.vae_config.get("downsample") * hf_config.latent_patch_size
                num_vae_tokens = (image_size.width // latent_downsample) * (image_size.height // latent_downsample)

            # For BAGEL, calculate number of tokens based on max patch size
            num_vit_tokens = hf_config.vit_max_num_patch_per_side**2
            # Use the image token ID from tokenizer
            return PromptUpdateDetails.select_token_id(
                seq=[img2img_vision_start]
                + [img2img_token_id] * num_vae_tokens
                + [img2img_vision_end]
                + [img2img_vision_start]
                + [img2img_token_id] * num_vit_tokens
                + [img2img_vision_end],
                embed_token_id=img2img_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_img2text_bagel,
            ),
            PromptReplacement(
                modality="img2img",
                target=[img2img_token_id],
                replacement=get_replacement_img2img_bagel,
            ),
        ]


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
    Omni version of BagelForConditionalGeneration.
    Currently just inherits from the upstream vLLM version.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        # VAE configuration
        self.latent_patch_size = getattr(config, "latent_patch_size", 2)
        self.downsample = config.vae_config.get("downsample")
        self.latent_downsample = self.downsample * self.latent_patch_size  # vae_downsample * patch_size
        self.max_latent_size = getattr(config, "max_latent_size", 32)
        self.latent_channel = config.vae_config.get("z_channels")

        hidden_size = config.llm_config.hidden_size
        patch_latent_dim = self.latent_patch_size**2 * self.latent_channel
        self.vae = VAEEncoder(default_ae_params())
        self.vae2llm = nn.Linear(patch_latent_dim, hidden_size)
        self.latent_pos_embed = PositionEmbedding(self.max_latent_size, hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)

    def get_kv_transfer_metadata(self, req_id: str) -> dict[str, Any] | None:
        """Return custom metadata for KV transfer.

        This method is called by KVTransferManager when transferring KV cache.
        """
        # [Omni] In the future, this should retrieve actual metadata from the model state based on req_id.
        # For now, we return a fixed value for testing purposes as requested.
        return {"ropes": [11]}

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Parse img2text (standard image)
        if any(k in kwargs for k in ("pixel_values", "image_embeds")):
            mm_input_by_modality["img2text"] = self._parse_and_validate_image_input(**kwargs)

        # Parse img2img
        # We check for specific img2img keys.
        # We map pixel_values_img2img -> pixel_values for the validator.
        img2img_keys = {"pixel_values_img2img": "pixel_values", "image_embeds_img2img": "image_embeds"}
        img2img_kwargs = {img2img_keys[k]: v for k, v in kwargs.items() if k in img2img_keys}

        if img2img_kwargs:
            # Construct kwargs for img2img validation by combining original kwargs
            # (which might contain necessary config/metadata) with remapped img2img data.
            # We assume the validator prefers the keys in img2img_kwargs.
            combined_kwargs = kwargs.copy()
            combined_kwargs.update(img2img_kwargs)
            mm_input_by_modality["img2img"] = self._parse_and_validate_image_input(**combined_kwargs)

        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "img2text":
                image_embeddings = self._process_img2text_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "img2img":
                img2img_embeddings = self._process_img2img_input(multimodal_input)
                multimodal_embeddings += tuple(img2img_embeddings)
        return multimodal_embeddings

    def get_flattened_position_ids(self, img_h, img_w, patch_size, max_num_patches_per_side):
        num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
        coords_h = torch.arange(0, num_patches_h)
        coords_w = torch.arange(0, num_patches_w)
        pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
        return pos_ids

    def _process_img2text_input(self, multimodal_input):
        return self._process_image_input(multimodal_input)

    def _process_img2img_input(self, multimodal_input):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        timestep = 0
        pixel_values = multimodal_input["pixel_values"]
        padded_latent = self.vae.encode(pixel_values)  # pixel_values shape: torch.Size([1, 3, 1024, 800])
        H, W = pixel_values.shape[2:]
        h = H // self.latent_downsample
        w = W // self.latent_downsample
        patchified_vae_latent_shapes.append((h, w))
        packed_timesteps = torch.tensor([timestep])
        packed_latent = list()
        p = self.latent_patch_size
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, : h * p, : w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)

        vae_position_ids = self.get_flattened_position_ids(
            pixel_values.shape[2],
            pixel_values.shape[3],
            self.latent_downsample,
            max_num_patches_per_side=self.max_latent_size,
        )
        packed_vae_position_ids.append(vae_position_ids)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            packed_timestep_embeds = self.time_embedder(packed_timesteps.to(padded_latent))
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        # Concatenate VAE latents and ViT embeddings along the sequence dimension
        # Resize pixel_values for ViT
        image_size = self.config.vit_config.image_size
        vit_pixel_values = torch.nn.functional.interpolate(
            pixel_values,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )
        vit_embeddings = self._process_image_input({"pixel_values": vit_pixel_values})
        if isinstance(vit_embeddings, (list, tuple)):
            vit_embeddings = vit_embeddings[0]

        combined_embeddings = torch.cat([packed_latent, vit_embeddings], dim=0)
        # Return as a list/tuple containing a single tensor for this single input item
        return [combined_embeddings]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for OmniBagelForConditionalGeneration.

        When VAE is disabled, we simply delegate to the parent class
        to ensure all standard mappings (including ViT) are handled correctly.
        When VAE is enabled, we extend the loading logic to handle VAE weights.
        """

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

            if "latent_pos_embed.pos_embed" in mapped_name and tensor.ndim == 2:
                npos, hdim = tensor.shape
                current_param = self.latent_pos_embed.pos_embed
                if current_param.shape != tensor.shape:
                    side = isqrt(int(npos))
                    if side * side == int(npos) and hdim == current_param.shape[1]:
                        current_param.data = current_param.data.new_empty((npos, hdim))
                        self.max_latent_size = int(side)
                        if hasattr(self.latent_pos_embed, "max_num_patch_per_side"):
                            self.latent_pos_embed.max_num_patch_per_side = int(side)

            filtered_weights.append((mapped_name, tensor))

        # Use the parent's mapper for standard components
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["vit_pos_embed.pos_embed"],
            ignore_unexpected_prefixes=["vae.", "latent_pos_embed.", "time_embedder.", "vae2llm."],
        )
        return loader.load_weights(filtered_weights, mapper=self.hf_to_vllm_mapper)
