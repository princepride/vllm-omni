from transformers import AutoConfig

from .bagel import BagelConfig

__all__ = ["BagelConfig"]

AutoConfig.register("bagel", BagelConfig)
