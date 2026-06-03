# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
LoRA injection: freeze backbone + wrap with PEFT + freeze extras.

Adapts to LoongForgeVLA four-layer architecture:
  - ModelFramework.architecture.backbone is the VLM backbone
  - Locates VLM interface via lora.vlm_module or auto-detection
"""
from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from loongforge.embodied.train.utils.peft.config import LoRASpec, is_lora_enabled

logger = logging.getLogger(__name__)

# VLM interface auto-detection registry
_VLM_ATTR_CANDIDATES = [
    "backbone",
    "qwen_vl_interface",
    "llama_vl_interface",
    "paligemma_vl_interface",
]


def _detect_vlm_interface(model: nn.Module):
    """Auto-detect VLM interface on model or model.architecture."""
    search_targets = [model]
    if hasattr(model, "architecture"):
        search_targets.append(model.architecture)

    for target in search_targets:
        for attr in _VLM_ATTR_CANDIDATES:
            iface = getattr(target, attr, None)
            if iface is not None and isinstance(iface, nn.Module):
                return iface
    return None


def apply_lora(
    model: nn.Module,
    cfg: Any,
    args: Any = None,
    *,
    print_summary: bool = True,
) -> nn.Module:
    """
    Apply LoRA in-place per spec. CLI args (when provided) override the YAML
    `lora:` block for `enabled` / `rank` / `alpha` / `target_modules`.

    Steps:
      1. Resolve VLM interface (from lora.vlm_module or auto-detect)
      2. Freeze ALL params of the VLM interface
      3. Inject PEFT LoRA layers (their params are trainable)
      4. Freeze each module in lora.freeze_extra_modules
      5. Other modules keep original requires_grad

    Returns the same model instance (mutated in place).
    """
    if not is_lora_enabled(args):
        return model

    from peft import get_peft_model

    spec = LoRASpec.from_args(args)
    lora_config = spec.peft_config()

    # 1. Resolve VLM interface
    if spec.vlm_module:
        vlm_interface = getattr(model, spec.vlm_module, None)
        if vlm_interface is None and hasattr(model, "architecture"):
            vlm_interface = getattr(model.architecture, spec.vlm_module, None)
        if vlm_interface is None:
            raise AttributeError(
                f"lora.vlm_module='{spec.vlm_module}' not found on model"
            )
    else:
        vlm_interface = _detect_vlm_interface(model)

    assert vlm_interface is not None, (
        "No VLM interface found for LoRA injection. "
        "Set lora.vlm_module explicitly in config."
    )

    # 2 + 3. Freeze backbone, inject PEFT
    for p in vlm_interface.parameters():
        p.requires_grad = False

    if hasattr(vlm_interface, "model") and isinstance(vlm_interface.model, nn.Module):
        vlm_interface.model = get_peft_model(vlm_interface.model, lora_config)
    else:
        vlm_interface = get_peft_model(vlm_interface, lora_config)

    # 4. Freeze extras
    for module_name in spec.freeze_extra_modules:
        extra = getattr(model, module_name, None)
        if extra is None and hasattr(model, "architecture"):
            extra = getattr(model.architecture, module_name, None)
        if extra is not None:
            n = sum(1 for _ in extra.parameters())
            for p in extra.parameters():
                p.requires_grad = False
            logger.info(f"Froze extra module '{module_name}' ({n} param tensors)")
        else:
            logger.warning(f"freeze_extra_modules: '{module_name}' not found, skipping")

    # 5. Logging
    if print_summary:
        logger.info("LoRA enabled on VLM backbone")
        try:
            if hasattr(vlm_interface, "model") and hasattr(vlm_interface.model, "print_trainable_parameters"):
                vlm_interface.model.print_trainable_parameters()
        except Exception:
            pass
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Total trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
            f"({100 * trainable / max(total, 1):.2f}%)"
        )

    return model
