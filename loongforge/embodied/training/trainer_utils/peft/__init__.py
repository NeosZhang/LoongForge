"""
LoRA / PEFT helpers for LoongForgeVLA.

Public API:
    is_lora_enabled(cfg)                          -> bool
    apply_lora(model, cfg)                        -> model (in-place)
    save_lora_checkpoint(accelerator, model, ...)
    load_and_merge(...)
"""
from training.trainer_utils.peft.config import LoRASpec, is_lora_enabled
from training.trainer_utils.peft.injector import apply_lora
from training.trainer_utils.peft.checkpoint import save_lora_checkpoint, load_and_merge

__all__ = [
    "LoRASpec",
    "is_lora_enabled",
    "apply_lora",
    "save_lora_checkpoint",
    "load_and_merge",
]
