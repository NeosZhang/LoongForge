# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Distributed checkpoint save/load/resume — replaces accelerator.save_state/load_state."""

import gc
import json
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .context import DistributedContext
from .parallel import unwrap_model

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    checkpoint_dir: str,
    ctx: DistributedContext,
    args,
):
    """Save checkpoint (rank0 for model weights, all ranks for FSDP optim state)."""
    path = os.path.join(checkpoint_dir, f"steps_{step}")
    ctx.barrier()

    state_dict = _get_full_state_dict(model, ctx)

    if ctx.is_main:
        os.makedirs(path, exist_ok=True)

        # Save model weights
        save_format = getattr(args, "save_format", "safetensors")
        if save_format == "safetensors":
            torch.cuda.empty_cache()
            gc.collect()
            _save_state_dict_safetensors(state_dict, os.path.join(path, "model.safetensors"))
            gc.collect()
            torch.cuda.empty_cache()
        else:
            torch.save(state_dict, os.path.join(path, "pytorch_model.pt"))

        # Resume metadata
        meta = {
            "completed_steps": step,
            "num_gpus": ctx.world_size,
        }
        with open(os.path.join(path, "resume_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Checkpoint saved: {path}")

    # Optionally save training state (optimizer + scheduler)
    if getattr(args, "save_training_state", False):
        _save_training_state(model, optimizer, scheduler, step, path, ctx)

    ctx.barrier()


def load_pretrained(model: nn.Module, checkpoint_path: str, ctx: DistributedContext) -> nn.Module:
    """Load pretrained weights (call BEFORE DDP/FSDP wrapping)."""
    if not checkpoint_path:
        return model

    resolved = _resolve_file(checkpoint_path)
    sd = _load_sd(resolved)

    # Filter out shape mismatches
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape != v.shape:
            skipped.append(k)
        else:
            filtered[k] = v

    if skipped and ctx.is_main:
        logger.warning(f"Skipped {len(skipped)} shape-mismatched keys")
        for k in skipped[:5]:
            logger.warning(f"  {k}")

    model.load_state_dict(filtered, strict=False)
    if ctx.is_main:
        logger.info(f"Loaded pretrained: {checkpoint_path}")
    return model


def get_latest_checkpoint(checkpoint_dir: str) -> Tuple[Optional[str], int]:
    """Find latest checkpoint directory."""
    if not os.path.isdir(checkpoint_dir):
        return None, 0

    candidates = []
    for d in os.listdir(checkpoint_dir):
        meta_path = os.path.join(checkpoint_dir, d, "resume_meta.json")
        if d.startswith("steps_") and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            candidates.append((d, meta["completed_steps"]))

    if not candidates:
        return None, 0

    candidates.sort(key=lambda x: x[1], reverse=True)
    latest_dir, latest_steps = candidates[0]
    return os.path.join(checkpoint_dir, latest_dir), latest_steps


def resume_training_state(model, optimizer, scheduler, checkpoint_path, ctx):
    """Load optimizer/scheduler state from checkpoint (call AFTER wrapping)."""
    state_file = os.path.join(checkpoint_path, "training_state.pt")
    if not os.path.exists(state_file):
        if ctx.is_main:
            logger.warning(f"No training_state.pt in {checkpoint_path}, warm restart")
        return

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    state = torch.load(state_file, map_location="cpu")

    if isinstance(model, FSDP):
        from torch.distributed.checkpoint.state_dict import set_state_dict, StateDictOptions
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_state_dict(model, optimizers=[optimizer],
                       model_state_dict={}, optim_state_dict={0: state["optimizer"]},
                       options=options)
    else:
        optimizer.load_state_dict(state["optimizer"])

    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    if ctx.is_main:
        logger.info(f"Training state resumed from {checkpoint_path}")


# ─── Internal Helpers ───


def _get_full_state_dict(model: nn.Module, ctx: DistributedContext) -> dict:
    """Get full state dict handling FSDP1/FSDP2/DDP."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP) or _is_fsdp2(model):
        from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
        options = StateDictOptions(full_state_dict=True, cpu_offload=False)
        model_sd, _ = get_state_dict(model, optimizers=[], options=options)
        return model_sd
    else:
        return unwrap_model(model).state_dict()


def _is_fsdp2(model: nn.Module) -> bool:
    """Check if model has FSDP2 (fully_shard) applied."""
    return hasattr(model, "_fsdp_state")


def _is_zero_optimizer(optimizer) -> bool:
    """Check if optimizer is a ZeroRedundancyOptimizer."""
    from torch.distributed.optim import ZeroRedundancyOptimizer
    return isinstance(optimizer, ZeroRedundancyOptimizer)


def _save_training_state(model, optimizer, scheduler, step, path, ctx):
    """Save optimizer + scheduler state."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP) or _is_fsdp2(model):
        from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        _, optim_sd = get_state_dict(model, optimizers=[optimizer], options=options)
    elif _is_zero_optimizer(optimizer):
        # ZeRO: consolidate sharded optimizer state to rank 0
        optimizer.consolidate_state_dict()
        optim_sd = optimizer.state_dict() if ctx.is_main else None
    else:
        optim_sd = optimizer.state_dict()

    if ctx.is_main:
        torch.save(
            {
                "optimizer": optim_sd,
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            os.path.join(path, "training_state.pt"),
        )


def _save_state_dict_safetensors(state_dict: dict, filepath: str):
    """Save state dict with safetensors, creating fresh tensors to avoid storage issues."""
    from safetensors.torch import save_file
    from torch.distributed.tensor import DTensor

    clean_sd = {}
    for k, v in state_dict.items():
        if isinstance(v, DTensor):
            v = v.full_tensor()
        clean_sd[k] = v.detach().cpu().clone()

    save_file(clean_sd, filepath)


def _resolve_file(checkpoint_path: str) -> str:
    if os.path.isdir(checkpoint_path):
        for name in ("model.safetensors", "pytorch_model.pt"):
            f = os.path.join(checkpoint_path, name)
            if os.path.exists(f):
                return f
        raise FileNotFoundError(f"No model file in {checkpoint_path}")
    return checkpoint_path


def _load_sd(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu")
