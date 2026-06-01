# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
LoRA checkpoint save / load+merge.

Replaces accelerator usage with native PyTorch + unwrap_model.

Checkpoint layout:
    <base_path>_lora_adapter/        <- PEFT adapter directory
    <base_path>_action_model.pt      <- non-VLM weights
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from loongforge.embodied.distributed.parallel import unwrap_model
from loongforge.embodied.train.utils.peft.injector import _detect_vlm_interface, _VLM_ATTR_CANDIDATES

logger = logging.getLogger(__name__)


def _resolve_vlm_interface(model: nn.Module, vlm_module: Optional[str] = None):
    """Return the VLM interface submodule."""
    if vlm_module:
        iface = getattr(model, vlm_module, None)
        if iface is None and hasattr(model, "architecture"):
            iface = getattr(model.architecture, vlm_module, None)
        assert iface is not None, f"VLM module '{vlm_module}' not found"
        return iface
    iface = _detect_vlm_interface(model)
    assert iface is not None, "No VLM interface found on model"
    return iface


def _vlm_attr_prefixes(model: nn.Module) -> tuple:
    """All VLM-interface attribute prefixes on model (for filtering state_dict)."""
    prefixes = []
    search_targets = [model]
    if hasattr(model, "architecture"):
        search_targets.append(model.architecture)
    for target in search_targets:
        for attr in _VLM_ATTR_CANDIDATES:
            if hasattr(target, attr):
                prefixes.append(f"{attr}.")
                if target is not model:
                    prefixes.append(f"architecture.{attr}.")
    return tuple(prefixes)


def save_lora_checkpoint(
    *,
    model: nn.Module,
    base_path: str,
    cfg: Any,
) -> None:
    """
    Save LoRA adapter + non-VLM weights.

    Creates:
      <base_path>_lora_adapter/        (PEFT adapter)
      <base_path>_action_model.pt      (non-VLM keys)
    """
    raw_model = unwrap_model(model)
    vlm_module = None
    if hasattr(cfg, "lora"):
        vlm_module = getattr(cfg.lora, "vlm_module", None) if hasattr(cfg.lora, "vlm_module") else cfg.lora.get("vlm_module", None)

    # 1. Adapter
    vlm_interface = _resolve_vlm_interface(raw_model, vlm_module)
    adapter_path = base_path + "_lora_adapter"
    if hasattr(vlm_interface, "model") and hasattr(vlm_interface.model, "save_pretrained"):
        vlm_interface.model.save_pretrained(adapter_path)
    elif hasattr(vlm_interface, "save_pretrained"):
        vlm_interface.save_pretrained(adapter_path)
    else:
        logger.warning("VLM interface has no save_pretrained method, saving raw state_dict")
        os.makedirs(adapter_path, exist_ok=True)
        torch.save(vlm_interface.state_dict(), os.path.join(adapter_path, "adapter_model.pt"))

    # 2. Non-VLM weights
    vlm_prefixes = _vlm_attr_prefixes(raw_model)
    state_dict = raw_model.state_dict()
    non_vlm_state = {
        k: v for k, v in state_dict.items() if not k.startswith(vlm_prefixes)
    }
    torch.save(non_vlm_state, base_path + "_action_model.pt")

    logger.info(
        f"LoRA checkpoint saved: {adapter_path} + non-VLM weights "
        f"({len(non_vlm_state)} keys)"
    )


def load_and_merge(
    *,
    base_model_factory: Callable[[], nn.Module],
    lora_adapter_dir: str,
    action_model_pt: str,
    output_path: str,
    vlm_module: Optional[str] = None,
) -> None:
    """
    Build base model, attach LoRA adapter, merge, load extras, save full ckpt.

    The output is a single .pt file for standard inference.
    """
    from peft import PeftModel

    print("[1/4] Build base model")
    model = base_model_factory()

    print(f"[2/4] Attach + merge LoRA adapter from {lora_adapter_dir}")
    vlm_interface = _resolve_vlm_interface(model, vlm_module)
    if hasattr(vlm_interface, "model"):
        vlm_interface.model = PeftModel.from_pretrained(vlm_interface.model, lora_adapter_dir)
        vlm_interface.model = vlm_interface.model.merge_and_unload()
    else:
        vlm_interface = PeftModel.from_pretrained(vlm_interface, lora_adapter_dir)
        vlm_interface = vlm_interface.merge_and_unload()
    print("  LoRA merged into VLM backbone")

    print(f"[3/4] Load non-VLM weights from {action_model_pt}")
    non_vlm_state = torch.load(action_model_pt, map_location="cpu")
    missing, unexpected = model.load_state_dict(non_vlm_state, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}...")
    print(
        f"  Loaded {len(non_vlm_state)} non-VLM keys "
        f"(missing {len(missing)} VLM keys as expected)"
    )

    print(f"[4/4] Save merged checkpoint to {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done! Merged checkpoint: {size_mb:.0f} MB")
