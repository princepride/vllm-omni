# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .magi_human_dit import DiTModel, MagiHumanDiTConfig
from .pipeline_magi_human import (
    MagiHumanPipeline,
    get_magi_human_post_process_func,
    get_magi_human_pre_process_func,
)

__all__ = [
    "MagiHumanPipeline",
    "DiTModel",
    "MagiHumanDiTConfig",
    "get_magi_human_post_process_func",
    "get_magi_human_pre_process_func",
]
