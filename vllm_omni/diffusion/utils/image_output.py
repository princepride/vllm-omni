# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from PIL import Image


def extract_images_from_outputs(outputs: Any) -> list[Image.Image]:
    """Extract PIL images from common Omni output wrappers."""
    images: list[Image.Image] = []

    for candidate in _iter_output_candidates(outputs):
        images.extend(_coerce_images(getattr(candidate, "images", None)))
        if images:
            return images

    for candidate in _iter_output_candidates(outputs):
        for mm_output in _iter_multimodal_outputs(candidate):
            image_payload = mm_output.get("image") or mm_output.get("images")
            images.extend(_coerce_images(image_payload))
            if images:
                return images

    return images


def _iter_output_candidates(value: Any, seen: set[int] | None = None) -> Iterable[Any]:
    seen = seen or set()
    if value is None:
        return

    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, Mapping):
        yield value
        for key in ("request_output", "outputs", "output", "payload"):
            if key in value:
                yield from _iter_output_candidates(value[key], seen)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_output_candidates(item, seen)
        return

    yield value

    unwrap = getattr(value, "unwrap", None)
    if callable(unwrap):
        unwrapped = unwrap()
        if unwrapped is not value:
            yield from _iter_output_candidates(unwrapped, seen)

    request_output = getattr(value, "request_output", None)
    if request_output is not None:
        yield from _iter_output_candidates(request_output, seen)

    nested_outputs = getattr(value, "outputs", None)
    if nested_outputs:
        yield from _iter_output_candidates(nested_outputs, seen)

    raw_output = getattr(value, "output", None)
    if raw_output is not None:
        yield from _iter_output_candidates(raw_output, seen)


def _iter_multimodal_outputs(candidate: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(candidate, Mapping):
        if "image" in candidate or "images" in candidate:
            yield candidate
        payload = candidate.get("payload")
        if isinstance(payload, Mapping):
            yield payload
        for key in ("multimodal_output", "_multimodal_output"):
            mm_output = candidate.get(key)
            if isinstance(mm_output, Mapping):
                yield mm_output
        return

    for attr in ("multimodal_output", "_multimodal_output"):
        mm_output = getattr(candidate, attr, None)
        if isinstance(mm_output, Mapping):
            yield mm_output


def _coerce_images(payload: Any) -> list[Image.Image]:
    if payload is None:
        return []
    if isinstance(payload, Image.Image):
        return [payload]
    if isinstance(payload, torch.Tensor):
        return _tensor_to_images(payload)
    if isinstance(payload, (list, tuple)):
        images: list[Image.Image] = []
        for item in payload:
            images.extend(_coerce_images(item))
        return images
    return []


def _tensor_to_images(tensor: torch.Tensor) -> list[Image.Image]:
    img = tensor.detach().to("cpu", dtype=torch.float32)
    if img.ndim == 4:
        images: list[Image.Image] = []
        for single in img:
            images.extend(_tensor_to_images(single))
        return images
    if img.ndim != 3:
        return []

    if img.shape[0] in (1, 3, 4):
        img = img.permute(1, 2, 0)

    if img.min().item() < 0.0:
        img = img / 2 + 0.5
    img = img.clamp(0, 1).mul(255).to(torch.uint8).contiguous().numpy()
    return [Image.fromarray(img)]
