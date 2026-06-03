# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
LoRA spec parsed from model_cfg `lora:` section.

Recognized fields:
    enabled              bool
    rank                 int    (default 32)
    alpha                int    (default 16)
    dropout              float  (default 0.05)
    target_modules       str | list[str]  (default "all-linear")
    init_lora_weights    str    (default "gaussian")
    vlm_module           str | None       (default None -> auto-detect)
    freeze_extra_modules str | list[str]  (default [])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def is_lora_enabled(args: Any) -> bool:
    """Return True iff LoRA is enabled via --lora CLI flag."""
    return bool(getattr(args, "lora", False))


@dataclass
class LoRASpec:
    """Backbone-agnostic LoRA application spec."""

    rank: int = 32
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Any = "all-linear"
    init_lora_weights: str = "gaussian"
    vlm_module: str | None = None
    freeze_extra_modules: list[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args: Any) -> "LoRASpec":
        """Build LoRASpec from CLI args (--lora-rank/--lora-alpha/--lora-target-modules)."""
        target = getattr(args, "lora_target_modules", "all-linear")
        if isinstance(target, str) and "," in target:
            target = [m.strip() for m in target.split(",") if m.strip()]

        return cls(
            rank=int(getattr(args, "lora_rank", 32)),
            alpha=int(getattr(args, "lora_alpha", 16)),
            target_modules=target,
        )

    def peft_config(self):
        """Build a peft.LoraConfig from this spec."""
        from peft import LoraConfig

        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            init_lora_weights=self.init_lora_weights,
        )
