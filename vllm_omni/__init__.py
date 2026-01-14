"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore
from transformers import AutoConfig
from vllm.model_executor.models import ModelRegistry

from vllm_omni.diffusion.models.bagel.bagel_transformer import BagelConfig
from vllm_omni.model_executor.models.bagel.bagel import BagelForConditionalGeneration

from .config import OmniModelConfig
from .entrypoints.async_omni import AsyncOmni

# Main entry points
from .entrypoints.omni import Omni

from .version import __version__, __version_tuple__  # isort:skip


AutoConfig.register("bagel", BagelConfig)
ModelRegistry.register_model("BagelForConditionalGeneration", BagelForConditionalGeneration)

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]
