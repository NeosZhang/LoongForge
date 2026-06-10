# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Distributed checkpoint save/load/resume — replaces accelerator.save_state/load_state."""

import gc
import json
import logging
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
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
    epoch: int = 0,
    dataloader_state: Optional[Dict] = None,
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
            "epoch": epoch,
        }
        with open(os.path.join(path, "resume_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Checkpoint saved: {path}")

    # Save training state (optimizer + scheduler + RNG) — required for true resume.
    # Can be disabled by --no-save-training-state for weights-only export.
    if getattr(args, "save_training_state", True):
        _save_training_state(
            model,
            optimizer,
            scheduler,
            epoch,
            path,
            ctx,
            args,
            dataloader_state=dataloader_state,
        )

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


def get_latest_checkpoint(
    checkpoint_dir: str, require_training_state: bool = True
) -> Tuple[Optional[str], int, int]:
    """Find latest *resumable* checkpoint directory.

    Picks the `steps_N` dir with the largest N by name parse, then validates
    only that one (resume_meta.json present, plus training_state.pt when
    `require_training_state`). No fallback to older dirs — if the latest is
    incomplete the caller should fix it explicitly.

    Returns: (path, completed_steps, epoch)
    """
    if not os.path.isdir(checkpoint_dir):
        return None, 0, 0

    steps = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("steps_") and d[len("steps_"):].isdigit():
            steps.append((int(d[len("steps_"):]), d))
    if not steps:
        return None, 0, 0
    _, latest_d = max(steps)

    ckpt_dir = os.path.join(checkpoint_dir, latest_d)
    meta_path = os.path.join(ckpt_dir, "resume_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"resume_meta.json not found in {ckpt_dir}")
    if require_training_state and not os.path.exists(
        os.path.join(ckpt_dir, "training_state.pt")
    ):
        raise FileNotFoundError(
            f"training_state.pt not found in {ckpt_dir}; cannot resume "
            f"(re-save with --save-training-state)."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    return ckpt_dir, int(meta["completed_steps"]), int(meta.get("epoch", 0))


def resume_training_state(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    ctx,
    args=None,
    restore_rng: bool = True,
) -> Tuple[Optional[int], Dict, Optional[list]]:
    """Load optimizer/scheduler/RNG state from checkpoint (call AFTER wrapping).

    Returns: (epoch index recorded at the saved step or None if not present,
              dataloader state, per-rank RNG state).
    """
    state_file = os.path.join(checkpoint_path, "training_state.pt")
    if not os.path.exists(state_file):
        raise FileNotFoundError(
            f"No training_state.pt in {checkpoint_path}; cannot resume training. "
            f"Re-save the checkpoint with --save-training-state (default on)."
        )

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    state = torch.load(state_file, map_location="cpu", weights_only=False)

    if isinstance(model, FSDP) or _is_fsdp2(model):
        from torch.distributed.checkpoint.state_dict import set_state_dict, StateDictOptions
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_state_dict(model, optimizers=[optimizer],
                       model_state_dict={}, optim_state_dict={0: state["optimizer"]},
                       options=options)
    else:
        optimizer.load_state_dict(state["optimizer"])
    if ctx.is_main:
        logger.info("optimizer resumed successfully")

    if "scheduler" in state and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
        if ctx.is_main:
            logger.info("scheduler resumed successfully")

    # Per-rank RNG state — packed inside training_state.pt under "rng_state_per_rank".
    # When restore_rng=False, defer restoration to the caller (which will invoke
    # `restore_rank_rng_state` later, after dataloader iter() init).
    rng_per_rank = None
    if restore_rng:
        restore_rank_rng_state(state.get("rng_state_per_rank"), ctx, source=state_file)
        if ctx.is_main:
            logger.info("RNG state resumed successfully")
    else:
        rng_per_rank = state.get("rng_state_per_rank")

    dataloader_state = {}
    dataloader_state_per_rank = state.get("dataloader_state_per_rank")
    if dataloader_state_per_rank is not None:
        if len(dataloader_state_per_rank) != ctx.world_size:
            raise RuntimeError(
                f"Dataloader state was saved for world_size={len(dataloader_state_per_rank)} "
                f"but current world_size={ctx.world_size}."
            )
        dataloader_state = dataloader_state_per_rank[ctx.rank] or {}

    saved_epoch = state.get("epoch")
    saved_epoch = int(saved_epoch) if saved_epoch is not None else None
    return saved_epoch, dataloader_state, rng_per_rank


def restore_rank_rng_state(rng_per_rank, ctx: DistributedContext, source: str = "checkpoint"):
    """Validate per-rank RNG payload and restore this rank's stream."""
    if rng_per_rank is None:
        raise KeyError(
            f"RNG state not present in {source} (older checkpoint format). "
            f"Re-save with --save-training-state."
        )
    if len(rng_per_rank) != ctx.world_size:
        raise RuntimeError(
            f"RNG state was saved for world_size={len(rng_per_rank)} but current "
            f"world_size={ctx.world_size}."
        )
    rng = rng_per_rank[ctx.rank]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng["torch_cuda"], device=ctx.device)


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


def _save_training_state(
    model,
    optimizer,
    scheduler,
    epoch,
    path,
    ctx,
    args=None,
    dataloader_state=None,
):
    """Save optimizer + scheduler + per-rank RNG state into a single file on rank0.

    RNG state is collected from every rank via all_gather_object so a resumed
    run with the same world_size can restore each rank's stream from the same
    training_state.pt (no separate per-rank files).
    """
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

    # Collect per-rank RNG. Each rank captures its own four streams; rank0 gathers
    # them into a list indexed by rank for embedding in the unified state file.
    local_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(ctx.device) if torch.cuda.is_available() else None,
    }
    if ctx.is_distributed:
        import torch.distributed as dist
        rng_per_rank = [None] * ctx.world_size
        dist.all_gather_object(rng_per_rank, local_rng)
    else:
        rng_per_rank = [local_rng]

    dataloader_state_per_rank = None
    if dataloader_state:
        if ctx.is_distributed:
            import torch.distributed as dist
            dataloader_state_per_rank = [None] * ctx.world_size
            dist.all_gather_object(dataloader_state_per_rank, dataloader_state)
        else:
            dataloader_state_per_rank = [dataloader_state]

    if ctx.is_main:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "optimizer": optim_sd,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "dataloader_state_per_rank": dataloader_state_per_rank,
                "rng_state_per_rank": rng_per_rank,
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
