"""
trainer_tools.py - Training utility collection

Provides:
  - build_param_lr_groups: Set different learning rates per module
  - TrainerUtils: Freeze/unfreeze, print parameters, load checkpoint, distributed preparation
  - normalize_dotlist_args: CLI argument normalization
  - is_main_process: Check if current process is main
"""

import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── color helpers (auto-off when not a tty) ──────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(t, *codes):
    return "\033[{}m{}\033[0m".format(";".join(map(str, codes)), t) if _USE_COLOR else str(t)


def _dim(t):         return _c(t, 2)
def _yellow(t):      return _c(t, 93)
def _cyan(t):        return _c(t, 96)
def _bold_green(t):  return _c(t, 1, 92)
def _bold_red(t):    return _c(t, 1, 91)
def _bold_yellow(t): return _c(t, 1, 93)
def _bold_cyan(t):   return _c(t, 1, 96)


# ═══════════════════════════════════════════════════════════════
# CLI arg normalization
# ═══════════════════════════════════════════════════════════════

def normalize_dotlist_args(args: List[str]) -> List[str]:
    """Convert ['--x.y', 'val'] and ['--flag'] -> ['x.y=val', 'flag=true']."""
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        elif "=" in arg:
            normalized.append(arg)
    return normalized


# ═══════════════════════════════════════════════════════════════
# Per-module learning rate groups
# ═══════════════════════════════════════════════════════════════

def build_param_lr_groups(model: nn.Module, cfg) -> List[Dict]:
    """
    Set different learning rates for different modules based on cfg.trainer.learning_rate.

    cfg.trainer.learning_rate structure example:
        base: 2.5e-05
        backbone: 1.0e-05
        action_model: 1.0e-04

    Also supports cfg.trainer.freeze_modules to freeze specified modules.
    """
    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)

    # Parse freeze patterns
    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    # Collect frozen param IDs
    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            logger.warning(f"freeze path not found: {freeze_path}")
            continue

    # Per-module learning rates
    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            params = [
                p for p in module.parameters()
                if id(p) not in frozen_params and p.requires_grad
            ]
            if params:
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            logger.warning(f"module path `{module_name}` not found in model")

    # Base learning rate for remaining params
    other_params = [
        p for p in model.parameters()
        if id(p) not in used_params and id(p) not in frozen_params and p.requires_grad
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════

def is_main_process() -> bool:
    """Check if current process is main."""
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


def _is_safetensors_path(path: str) -> bool:
    path = str(path)
    if path.endswith(".safetensors"):
        return True
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.safetensors")):
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# TrainerUtils class
# ═══════════════════════════════════════════════════════════════

class TrainerUtils:
    """Training utility static method collection."""

    @staticmethod
    def freeze_backbones(model: nn.Module, freeze_modules: str = "") -> nn.Module:
        """
        Freeze specified submodules based on comma-separated module path list.

        Args:
            model: nn.Module
            freeze_modules: Comma-separated paths like "backbone,condition"
        """
        frozen = []
        if not freeze_modules or not isinstance(freeze_modules, str):
            return model

        if is_main_process():
            logger.info(f"freeze_modules: {freeze_modules}")

        patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]
        for path in patterns:
            attrs = path.split(".")
            module = model
            try:
                for attr in attrs:
                    module = getattr(module, attr)
                for param in module.parameters():
                    param.requires_grad = False
                frozen.append(path)
            except AttributeError:
                logger.warning(f"module path not found, skipping freeze: {path}")
                continue

        if is_main_process():
            logger.info(f"frozen: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model: nn.Module) -> Optional[Tuple[int, int]]:
        """Print model parameter statistics."""
        if dist.is_initialized() and dist.get_rank() != 0:
            return None
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Total {num_params / 1e6:.2f}M  "
            f"Trainable {num_trainable / 1e6:.2f}M  "
            f"({100 * num_trainable / max(num_params, 1):.1f}%)"
        )
        return num_params, num_trainable

    @staticmethod
    def load_pretrained_checkpoint(
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        reload_modules: Optional[str] = None,
    ) -> nn.Module:
        """
        Load pretrained checkpoint.

        Supports:
          - safetensors / pt format
          - Directory format (auto-finds model.safetensors or pytorch_model.pt)
          - Full loading (skips shape-mismatched keys)
          - Partial loading by module (reload_modules="action_model,condition")
        """
        if not checkpoint_path:
            return model
        if is_main_process():
            logger.info(f"loading checkpoint: {checkpoint_path}")

        # Resolve directory to file
        resolved = checkpoint_path
        if os.path.isdir(checkpoint_path):
            sf_path = os.path.join(checkpoint_path, "model.safetensors")
            pt_path = os.path.join(checkpoint_path, "pytorch_model.pt")
            if os.path.exists(sf_path):
                resolved = sf_path
            elif os.path.exists(pt_path):
                resolved = pt_path
            else:
                raise RuntimeError(
                    f"checkpoint directory missing model.safetensors or pytorch_model.pt: "
                    f"{checkpoint_path}"
                )

        # Load state dict
        try:
            if _is_safetensors_path(resolved):
                from safetensors.torch import load_file
                sf_path = str(checkpoint_path)
                if os.path.isdir(sf_path):
                    sf_path = os.path.join(sf_path, "model.safetensors")
                checkpoint = load_file(sf_path)
            else:
                checkpoint = torch.load(resolved, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"loading checkpoint failed: {e}")

        if reload_modules:
            # Partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                parts = path.split(".")
                module = model
                try:
                    for part in parts:
                        module = getattr(module, part)
                    prefix = path + "."
                    sub_state = {
                        k[len(prefix):]: v
                        for k, v in checkpoint.items()
                        if k.startswith(prefix)
                    }
                    if sub_state:
                        module.load_state_dict(sub_state, strict=True)
                        if is_main_process():
                            logger.info(f"loaded module '{path}'")
                    else:
                        logger.warning(f"no keys found for module '{path}'")
                except AttributeError:
                    logger.error(f"module path not found: '{path}'")
        else:
            # Full load with shape filtering
            model_state = model.state_dict()
            filtered = {}
            skipped = []
            for k, v in checkpoint.items():
                if k in model_state and model_state[k].shape != v.shape:
                    skipped.append(
                        f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape)}"
                    )
                else:
                    filtered[k] = v
            if skipped and is_main_process():
                logger.warning(f"skipped {len(skipped)} shape-mismatched keys")
                for s in skipped[:5]:
                    logger.warning(f"  {s}")
            model.load_state_dict(filtered, strict=False)
            if is_main_process():
                logger.info("loaded full model parameters")

        return model

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """Wrap distributed training components using Accelerator.prepare."""
        prepared = accelerator.prepare(*components)
        # For DDP with parameter reuse, set static_graph
        comps = prepared if isinstance(prepared, tuple) else (prepared,)
        for comp in comps:
            if hasattr(comp, "module") and hasattr(comp, "_set_static_graph"):
                comp._set_static_graph()
        return prepared

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Tuple[Optional[str], int]:
        """
        Find the latest resumable checkpoint.

        Returns (checkpoint_path, completed_steps) or (None, 0).
        Supports steps_N/ directory format (with resume_meta.json).
        """
        if not os.path.isdir(checkpoint_dir):
            return None, 0

        candidates = []
        for d in sorted(os.listdir(checkpoint_dir)):
            if d.startswith("steps_") and os.path.isdir(os.path.join(checkpoint_dir, d)):
                meta_path = os.path.join(checkpoint_dir, d, "resume_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    candidates.append((d, meta["completed_steps"], meta))

        if not candidates:
            return None, 0

        candidates.sort(key=lambda x: x[1], reverse=True)
        latest_dir, latest_steps, _ = candidates[0]
        return os.path.join(checkpoint_dir, latest_dir), latest_steps
