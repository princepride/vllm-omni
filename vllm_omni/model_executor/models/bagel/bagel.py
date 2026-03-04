from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from math import isqrt
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
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

logger = init_logger(__name__)


class OmniBagelProcessor(BagelProcessor):
    def __call__(self, text=None, images=None, **kwargs):
        is_img2img = kwargs.pop("is_img2img", False)

        if is_img2img and images is not None:
            image_kwargs = kwargs.copy()
            image_kwargs["do_resize"] = False
            image_kwargs["do_rescale"] = True
            if "return_tensors" not in image_kwargs:
                image_kwargs["return_tensors"] = "pt"

            pixel_values = self.image_processor(images, **image_kwargs)

            text_inputs = self.tokenizer(text, **kwargs) if text is not None else None

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
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1, "img2img": 1}

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(OmniBagelProcessor, **kwargs)

    def get_hf_config(self):
        config = super().get_hf_config()
        if not getattr(self, "_latent_size_patched", False):
            self._latent_size_patched = True
            self._patch_max_latent_size(config)
        return config

    def _patch_max_latent_size(self, config):
        """Infer correct max_latent_size from the model's latent_pos_embed
        weight, since the HF config value may be stale (e.g. 32 vs 64)."""
        import json
        from pathlib import Path

        model_name = self.ctx.model_config.model
        try:
            p = Path(model_name)
            if p.is_dir():
                index_path = p / "model.safetensors.index.json"
            else:
                from huggingface_hub import hf_hub_download

                index_path = Path(hf_hub_download(model_name, "model.safetensors.index.json"))

            if not index_path.exists():
                return

            with open(index_path) as f:
                index = json.load(f)

            shard = index.get("weight_map", {}).get("latent_pos_embed.pos_embed")
            if not shard:
                return

            from safetensors import safe_open

            with safe_open(str(index_path.parent / shard), framework="pt") as f:
                if "latent_pos_embed.pos_embed" in f.keys():
                    npos = f.get_slice("latent_pos_embed.pos_embed").get_shape()[0]
                    side = isqrt(npos)
                    if side * side == npos:
                        old = getattr(config, "max_latent_size", 32)
                        if old != side:
                            config.max_latent_size = side
                            logger.info(
                                "[Processor] Patched max_latent_size: %d -> %d (from latent_pos_embed shape[0]=%d)",
                                old,
                                side,
                                npos,
                            )
        except Exception as e:
            logger.warning("[Processor] Could not infer max_latent_size: %s", e)

    def get_data_parser(self) -> "OmniBagelDataParser":
        return OmniBagelDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )


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
        return {"pixel_values_img2img": self.get_all()}


class OmniBagelDataParser(MultiModalDataParser):
    def _parse_img2img_data(self, data: ModalityData) -> ModalityDataItems | None:
        items = self._parse_image_data(data)
        if items is None:
            return None
        return Img2ImgProcessorItems(items.data)

    def _get_subparsers(self):
        parsers = super()._get_subparsers()
        parsers["img2img"] = self._parse_img2img_data
        return parsers


class OmniBagelMultiModalProcessor(BaseMultiModalProcessor[OmniBagelProcessingInfo]):
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

        if has_image and has_img2img:
            outputs = BatchFeature()

            img_data = dict(mm_data)
            if "pixel_values_img2img" in img_data:
                del img_data["pixel_values_img2img"]
            kwargs_img = dict(mm_kwargs)
            kwargs_img["is_img2img"] = False
            out_img = super()._call_hf_processor(prompt, img_data, kwargs_img, tok_kwargs)
            if "pixel_values" in out_img:
                outputs["pixel_values"] = out_img["pixel_values"]
            for k, v in out_img.items():
                if k != "pixel_values":
                    outputs[k] = v

            img2img_data = dict(mm_data)
            if "images" in img2img_data:
                del img2img_data["images"]
            img2img_data["images"] = img2img_data.pop("pixel_values_img2img")
            kwargs_img2img = dict(mm_kwargs)
            kwargs_img2img["is_img2img"] = True
            out_img2img = super()._call_hf_processor(prompt, img2img_data, kwargs_img2img, tok_kwargs)
            if "pixel_values" in out_img2img:
                outputs["pixel_values_img2img"] = out_img2img["pixel_values"]
            for k, v in out_img2img.items():
                if k not in outputs:
                    outputs[k] = v

            return outputs

        elif has_img2img:
            mm_data = dict(mm_data)
            mm_data["images"] = mm_data.pop("pixel_values_img2img")
            mm_kwargs = dict(mm_kwargs)
            mm_kwargs["is_img2img"] = True
            outputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
            if "pixel_values" in outputs:
                outputs["pixel_values_img2img"] = outputs.pop("pixel_values")
            return outputs

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
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        replacements: list[PromptReplacement] = []

        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        if image_token_id is not None:
            num_patches = hf_config.vit_max_num_patch_per_side**2

            def get_image_replacement(item_idx: int):
                return [image_token_id] * num_patches

            replacements.append(
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_image_replacement,
                )
            )

        img2img_token_id = tokenizer.get_vocab().get("<|fim_middle|>")
        if img2img_token_id is not None:
            vit_config = hf_config.vit_config
            image_size = vit_config.image_size
            num_vit_patches = (image_size // vit_config.patch_size) ** 2

            latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
            downsample = hf_config.vae_config.get("downsample", 8)
            latent_downsample = downsample * latent_patch_size

            def get_img2img_replacement(item_idx: int):
                h, w = image_size, image_size
                if "img2img" in mm_items:
                    item = mm_items.get_items("img2img", (Img2ImgProcessorItems, ImageEmbeddingItems))
                    if hasattr(item, "get_image_size"):
                        size = item.get_image_size(item_idx)
                        h, w = size.height, size.width

                max_latent_size = getattr(hf_config, "max_latent_size", 32)
                max_img_size = int(max_latent_size * latent_downsample)
                stride = latent_downsample
                scale = min(max_img_size / max(h, w), 1.0)
                min_img_size = min(256, max_img_size)
                scale = max(scale, min_img_size / min(h, w))
                new_h = max(stride, int(round(h * scale / stride) * stride))
                new_w = max(stride, int(round(w * scale / stride) * stride))
                new_h = min(new_h, max_img_size)
                new_w = min(new_w, max_img_size)

                num_vae_patches = (new_h // latent_downsample) * (new_w // latent_downsample)
                total = (num_vae_patches + 2) + (num_vit_patches + 2)
                return [img2img_token_id] * total

            replacements.append(
                PromptReplacement(
                    modality="img2img",
                    target=[img2img_token_id],
                    replacement=get_img2img_replacement,
                )
            )

        return replacements


class VAEEncoder(nn.Module):
    """Lightweight VAE encoder (no decoder) for embedding images in the AR stage."""

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

    Extends the base model with a VAE encoder so that img2img can embed
    both VAE latents and ViT features within the AR stage, producing a
    combined KV cache that is then transferred to the DiT stage.

    Position IDs are adjusted so that:
      - VAE tokens all share position 0
      - ViT tokens all share position 1
      - Text tokens use sequential positions starting from 2
    This matches the position scheme used by the single-stage DiT pipeline,
    ensuring the transferred KV cache + ropes are directly compatible with
    the DiT's denoising loop.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.latent_patch_size = getattr(config, "latent_patch_size", 2)
        self.downsample = config.vae_config.get("downsample")
        self.latent_downsample = self.downsample * self.latent_patch_size
        self.max_latent_size = getattr(config, "max_latent_size", 32)
        self.latent_channel = config.vae_config.get("z_channels")

        hidden_size = config.llm_config.hidden_size
        patch_latent_dim = self.latent_patch_size**2 * self.latent_channel
        self.vae = VAEEncoder(default_ae_params())
        self.vae2llm = nn.Linear(patch_latent_dim, hidden_size)
        self.latent_pos_embed = PositionEmbedding(self.max_latent_size, hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)

        self._pending_img2img_info: list[tuple[int, int, int, int]] = []
        self._ropes_queue: deque[dict[str, Any]] = deque()
        # Cached info from the most recent _process_img2img_input call, reused
        # by CFG companion forward passes (cfg_text / cfg_img) whose encoder
        # results are deduplicated and therefore don't trigger a second call.
        self._cached_img2img_info: tuple[int, int, int, int] | None = None
        self._cfg_companions_remaining: int = 0

        # Resolve <|vision_start|> / <|vision_end|> token IDs so the AR stage
        # can wrap VAE and ViT embeddings with the same boundary markers used
        # by the single-stage DiT pipeline.
        from transformers import AutoTokenizer

        tok_name = getattr(vllm_config.model_config, "tokenizer", None) or vllm_config.model_config.model
        _tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        for t in ["<|vision_start|>", "<|vision_end|>"]:
            if t not in _tok.get_vocab():
                _tok.add_tokens([t])
        self._start_of_image_id = int(_tok.convert_tokens_to_ids("<|vision_start|>"))
        self._end_of_image_id = int(_tok.convert_tokens_to_ids("<|vision_end|>"))
        logger.info(
            "[AR init] vision boundary token IDs: start=%d, end=%d", self._start_of_image_id, self._end_of_image_id
        )

    def _resize_to_stride(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Resize pixel values to stride-aligned dimensions
        (matches DiT's ``_resize_images_to_stride``)."""
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        stride = self.latent_downsample
        max_img_size = int(self.max_latent_size * stride)

        scale = min(max_img_size / max(H, W), 1.0)
        min_img_size = min(256, max_img_size)
        scale = max(scale, min_img_size / min(H, W))
        new_H = max(stride, int(round(H * scale / stride) * stride))
        new_W = max(stride, int(round(W * scale / stride) * stride))
        new_H = min(new_H, max_img_size)
        new_W = min(new_W, max_img_size)

        if new_H != H or new_W != W:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(new_H, new_W), mode="bicubic", align_corners=False
            )
        return pixel_values

    def get_kv_transfer_metadata(self, req_id: str) -> dict[str, Any] | None:
        if self._ropes_queue:
            meta = self._ropes_queue.popleft()
            logger.info("[AR kv_meta] req_id=%s -> %s", req_id, meta)
            return meta
        logger.info("[AR kv_meta] req_id=%s -> queue empty, returning None", req_id)
        return None

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        if any(k in kwargs for k in ("pixel_values", "image_embeds")):
            mm_input_by_modality["img2text"] = self._parse_and_validate_image_input(**kwargs)

        img2img_keys = {"pixel_values_img2img": "pixel_values", "image_embeds_img2img": "image_embeds"}
        img2img_kwargs = {img2img_keys[k]: v for k, v in kwargs.items() if k in img2img_keys}

        if img2img_kwargs:
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
        pixel_values = multimodal_input["pixel_values"]
        if pixel_values.ndim == 5:
            b, n, c, h, w = pixel_values.shape
            pixel_values = pixel_values.reshape(b * n, c, h, w)

        num_images = pixel_values.shape[0]
        image_size = self.config.vit_config.image_size
        p = self.latent_patch_size
        timestep = 0

        if self._ropes_queue:
            logger.info("[AR img2img] Clearing stale _ropes_queue (%d entries)", len(self._ropes_queue))
            self._ropes_queue.clear()

        logger.info(
            "[AR img2img] num_images=%d, input pixel_values shape=%s, dtype=%s, range=[%.4f, %.4f]",
            num_images,
            list(pixel_values.shape),
            pixel_values.dtype,
            pixel_values.min().item(),
            pixel_values.max().item(),
        )

        vit_pixel_values = torch.nn.functional.interpolate(
            pixel_values,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )
        logger.info(
            "[AR img2img] ViT pixel_values after resize: shape=%s, range=[%.4f, %.4f]",
            list(vit_pixel_values.shape),
            vit_pixel_values.min().item(),
            vit_pixel_values.max().item(),
        )

        vit_embeddings_tuple = self._process_image_input({"pixel_values": vit_pixel_values})
        logger.info(
            "[AR img2img] ViT embeddings: num=%d, shape[0]=%s, first[0,:5]=%s",
            len(vit_embeddings_tuple),
            list(vit_embeddings_tuple[0].shape),
            vit_embeddings_tuple[0][0, :5].tolist(),
        )

        # Embed <|vision_start|> / <|vision_end|> boundary markers
        marker_ids = torch.tensor(
            [self._start_of_image_id, self._end_of_image_id],
            device=pixel_values.device,
            dtype=torch.long,
        )
        marker_embeds = self.language_model.model.embed_tokens(marker_ids)
        start_embed = marker_embeds[0:1]  # [1, hidden_size]
        end_embed = marker_embeds[1:2]  # [1, hidden_size]

        results = []

        for i in range(num_images):
            single_pv = pixel_values[i : i + 1]
            orig_H, orig_W = single_pv.shape[2:]
            single_pv = self._resize_to_stride(single_pv)
            H, W = single_pv.shape[2:]
            logger.info(
                "[AR img2img][%d] stride resize: (%d,%d)->(%d,%d), pv range=[%.4f, %.4f]",
                i,
                orig_H,
                orig_W,
                H,
                W,
                single_pv.min().item(),
                single_pv.max().item(),
            )

            padded_latent = self.vae.encode(single_pv)
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            logger.info(
                "[AR img2img][%d] VAE encode: padded_latent shape=%s, latent grid h=%d w=%d, first[0,0,:5]=%s",
                i,
                list(padded_latent.shape),
                h,
                w,
                padded_latent[0, 0, 0, :5].tolist(),
            )

            latent = padded_latent[0][:, : h * p, : w * p]
            latent = latent.reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            logger.info(
                "[AR img2img][%d] patchified latent: shape=%s, first[:3]=%s",
                i,
                list(latent.shape),
                latent[0, :3].tolist(),
            )

            vae_position_ids = self.get_flattened_position_ids(
                H,
                W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            pos_embed = self.latent_pos_embed([vae_position_ids])
            packed_timesteps = torch.tensor([timestep], device=padded_latent.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                timestep_embeds = self.time_embedder(packed_timesteps.to(padded_latent))
            vae_embeds = self.vae2llm(latent) + timestep_embeds + pos_embed
            logger.info(
                "[AR img2img][%d] vae_embeds: shape=%s, first[0,:5]=%s",
                i,
                list(vae_embeds.shape),
                vae_embeds[0, :5].tolist(),
            )

            vit_emb = vit_embeddings_tuple[i] if i < len(vit_embeddings_tuple) else vit_embeddings_tuple[0]

            # Match single-stage DiT structure:
            #   [start_of_image, vae_1..vae_N, end_of_image,
            #    start_of_image, vit_1..vit_M, end_of_image]
            se = start_embed.to(vae_embeds.dtype)
            ee = end_embed.to(vae_embeds.dtype)
            combined = torch.cat([se, vae_embeds, ee, se, vit_emb, ee], dim=0)
            results.append(combined)

            num_vae = h * w + 2  # +2 for start/end markers
            num_vit = vit_emb.shape[0] + 2
            info = (num_vae, num_vit, int(H), int(W))
            self._pending_img2img_info.append(info)
            self._cached_img2img_info = info
            self._cfg_companions_remaining = 2  # cfg_text + cfg_img
            logger.info(
                "[AR img2img][%d] combined: shape=%s "
                "(num_vae=%d incl markers, num_vit=%d incl markers), "
                "image_shape=(%d,%d)",
                i,
                list(combined.shape),
                num_vae,
                num_vit,
                H,
                W,
            )

        return tuple(results)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if self._pending_img2img_info:
            logger.info(
                "[AR forward] before adjust: input_ids shape=%s, "
                "positions shape=%s, inputs_embeds shape=%s, "
                "pending_info=%s",
                list(input_ids.shape) if input_ids is not None else None,
                list(positions.shape),
                list(inputs_embeds.shape) if inputs_embeds is not None else None,
                self._pending_img2img_info,
            )
            if input_ids is not None:
                logger.info("[AR forward] input_ids[:30]=%s", input_ids[:30].tolist())
            positions = self._adjust_positions_for_img2img(positions)

        elif self._cfg_companions_remaining > 0 and self._cached_img2img_info is not None:
            self._cfg_companions_remaining -= 1
            cached = self._cached_img2img_info
            num_vae, num_vit, img_H, img_W = cached
            num_img2img = num_vae + num_vit
            seq_len = inputs_embeds.shape[0] if inputs_embeds is not None else positions.shape[0]

            if inputs_embeds is not None and seq_len >= num_img2img:
                logger.info("[AR forward] cfg_text companion: seq_len=%d, reusing cached info=%s", seq_len, cached)
                self._pending_img2img_info = [cached]
                positions = self._adjust_positions_for_img2img(positions)
            else:
                rope = int(positions[seq_len - 1].item()) + 1
                self._ropes_queue.append({"ropes": [rope]})
                logger.info("[AR forward] cfg_img companion: seq_len=%d, rope=%d", seq_len, rope)

            if self._cfg_companions_remaining == 0:
                self._cached_img2img_info = None

        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def _adjust_positions_for_img2img(self, positions: torch.Tensor) -> torch.Tensor:
        """Rewrite position IDs to match the single-stage DiT scheme:
        VAE tokens -> position 0, ViT tokens -> position 1, text -> 2, 3, ...
        Also pushes per-request ropes + image_shape to the FIFO consumed by
        ``get_kv_transfer_metadata``.
        """
        info_list = self._pending_img2img_info
        self._pending_img2img_info = []

        if not info_list:
            return positions

        logger.info("[AR positions] original positions[:20]=%s ... total=%d", positions[:20].tolist(), len(positions))

        boundaries = [0]
        for i in range(1, len(positions)):
            if positions[i] < positions[i - 1]:
                boundaries.append(i)
        boundaries.append(len(positions))

        num_requests = len(boundaries) - 1
        new_positions = positions.clone()

        logger.info("[AR positions] %d requests, boundaries=%s, img2img_info=%s", num_requests, boundaries, info_list)

        img2img_idx = 0
        for req_idx in range(num_requests):
            start = boundaries[req_idx]
            end = boundaries[req_idx + 1]
            req_len = end - start

            if img2img_idx < len(info_list):
                num_vae, num_vit, img_H, img_W = info_list[img2img_idx]
                num_img2img = num_vae + num_vit

                if req_len >= num_img2img:
                    new_positions[start : start + num_vae] = 0
                    new_positions[start + num_vae : start + num_img2img] = 1
                    num_text = req_len - num_img2img
                    if num_text > 0:
                        new_positions[start + num_img2img : end] = torch.arange(
                            2, 2 + num_text, device=positions.device, dtype=positions.dtype
                        )

                    rope = 2 + num_text
                    self._ropes_queue.append(
                        {
                            "ropes": [rope],
                            "image_shape": [img_H, img_W],
                        }
                    )
                    logger.info(
                        "[AR positions] req[%d]: img2img, len=%d, "
                        "num_vae=%d, num_vit=%d, num_text=%d, "
                        "rope=%d, image_shape=(%d,%d), "
                        "positions[start:start+5]=%s",
                        req_idx,
                        req_len,
                        num_vae,
                        num_vit,
                        num_text,
                        rope,
                        img_H,
                        img_W,
                        new_positions[start : start + 5].tolist(),
                    )
                    img2img_idx += 1
                    continue

            rope = int(new_positions[end - 1].item()) + 1
            self._ropes_queue.append({"ropes": [rope]})
            logger.info("[AR positions] req[%d]: text-only, len=%d, rope=%d", req_idx, req_len, rope)

        return new_positions

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        generation_keywords_to_skip = [
            "moe_gen",
            "llm2vae",
            "decoder.",
        ]

        def _map_vae_weight_name(name: str) -> str:
            if name.startswith("encoder."):
                return "vae." + name
            if name.startswith("reg."):
                return "vae." + name
            return name

        filtered_weights = []
        for name, tensor in weights:
            if any(skip in name for skip in generation_keywords_to_skip):
                continue

            mapped_name = _map_vae_weight_name(name)

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
                        setattr(self.config, "max_latent_size", int(side))
                        if hasattr(self.latent_pos_embed, "max_num_patch_per_side"):
                            self.latent_pos_embed.max_num_patch_per_side = int(side)
                        logger.info(
                            "[AR load_weights] Updated max_latent_size to %d (from latent_pos_embed shape %s)",
                            side,
                            list(tensor.shape),
                        )

            filtered_weights.append((mapped_name, tensor))

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["vit_pos_embed.pos_embed"],
            ignore_unexpected_prefixes=["vae.", "latent_pos_embed.", "time_embedder.", "vae2llm."],
        )
        return loader.load_weights(filtered_weights, mapper=self.hf_to_vllm_mapper)
