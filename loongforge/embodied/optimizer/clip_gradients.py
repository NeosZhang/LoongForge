# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Gradient clipping and NaN cleaning."""

import torch
import torch.nn as nn


def clip_gradients(model: nn.Module, max_norm: float):
    """Gradient clipping for FSDP with mixed-dtype gradients (fp32 + bf16).

    FSDP shards parameters across ranks, so each rank only holds a shard of
    gradients. We compute local norm in float32, all-reduce to get global norm,
    then clip.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    is_fsdp = isinstance(model, FSDP) or hasattr(model, "_fsdp_state")
    if not is_fsdp:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return

    # Compute local sharded norm in float32 (handles mixed dtype)
    local_norm_sq = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            local_norm_sq += p.grad.detach().float().norm(2) ** 2

    # All-reduce to get global norm across all ranks
    torch.distributed.all_reduce(local_norm_sq)
    total_norm = local_norm_sq.sqrt()

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(clip_coef.to(p.grad.dtype))


def clean_nan_gradients(model: nn.Module):
    """Replace NaN/Inf gradients with 0."""
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)
