from vllm.model_executor.models.bagel import BagelForConditionalGeneration as _BagelForConditionalGeneration


class BagelForConditionalGeneration(_BagelForConditionalGeneration):
    """
    Omni version of BagelForConditionalGeneration.
    Currently just inherits from the upstream vLLM version.
    """

    pass
    # def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
    #     mm_input_by_modality = {}

    #     # Parse img2text (standard image)
    #     if any(k in kwargs for k in ("pixel_values", "image_embeds")):
    #         mm_input_by_modality["img2text"] = self._parse_and_validate_image_input(**kwargs)

    #     # Parse img2img
    #     # We check for specific img2img keys.
    #     # We map pixel_values_img2img -> pixel_values for the validator.
    #     img2img_keys = {"pixel_values_img2img": "pixel_values", "image_embeds_img2img": "image_embeds"}
    #     img2img_kwargs = {img2img_keys[k]: v for k, v in kwargs.items() if k in img2img_keys}

    #     if img2img_kwargs:
    #         # Construct kwargs for img2img validation by combining original kwargs
    #         # (which might contain necessary config/metadata) with remapped img2img data.
    #         # We assume the validator prefers the keys in img2img_kwargs.
    #         combined_kwargs = kwargs.copy()
    #         combined_kwargs.update(img2img_kwargs)
    #         mm_input_by_modality["img2img"] = self._parse_and_validate_image_input(**combined_kwargs)

    #     return mm_input_by_modality

    # def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
    #     mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
    #     if not mm_input_by_modality:
    #         return None

    #     # The result multimodal_embeddings is tuple of tensors, with each
    #     # tensor correspoending to a multimodal data item (image or video).
    #     multimodal_embeddings: tuple[torch.Tensor, ...] = ()

    #     # NOTE: It is important to iterate over the keys in this dictionary
    #     # to preserve the order of the modalities.
    #     for modality in mm_input_by_modality:
    #         multimodal_input = mm_input_by_modality[modality]
    #         if modality == "img2text":
    #             image_embeddings = self._process_img2text_input(multimodal_input)
    #             multimodal_embeddings += tuple(image_embeddings)
    #         if modality == "img2img":
    #             img2img_embeddings = self._process_img2img_input(multimodal_input)
    #             multimodal_embeddings += tuple(img2img_embeddings)
    #     return multimodal_embeddings

    # def _process_img2text_input(self, multimodal_input):
    #     return self._process_image_input(multimodal_input)

    # def _process_img2img_input(self, multimodal_input):
    #     # As requested, temporarily use img2text's processing
    #     return self._process_img2text_input(multimodal_input)
