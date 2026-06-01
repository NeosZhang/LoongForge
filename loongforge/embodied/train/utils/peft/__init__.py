# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
LoRA / PEFT helpers for LoongForge Embodied.

Public API:
    is_lora_enabled(cfg)                 -> bool
    apply_lora(model, cfg)               -> model (in-place)
    save_lora_checkpoint(model, ...)
    load_and_merge(...)
"""
from loongforge.embodied.train.utils.peft.config import LoRASpec, is_lora_enabled
from loongforge.embodied.train.utils.peft.injector import apply_lora
from loongforge.embodied.train.utils.peft.checkpoint import save_lora_checkpoint, load_and_merge

__all__ = [
    "LoRASpec",
    "is_lora_enabled",
    "apply_lora",
    "save_lora_checkpoint",
    "load_and_merge",
]
