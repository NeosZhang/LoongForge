# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Gradient clipping and NaN cleaning."""

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
import torch.distributed as dist
def get_grad_norm(model: nn.Module) -> float:
    """Compute global gradient norm, accounting for FSDP sharding.

    For FSDP models, gradients are sharded across ranks. Each rank computes
    its local norm squared, then all-reduce sums them to get the global norm.
    For non-FSDP models, computes the norm directly.

    Args:
        model: The model whose gradients to analyze. Can be a vanilla PyTorch
            module, FSDP-wrapped module, or module with FSDP sub-modules.

    Returns:
        The L2 norm of all model gradients (global norm for distributed).
    """

    is_fsdp = isinstance(model, FSDPModule)

    total_norm_sq = torch.zeros((), device=next(model.parameters()).device)

    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            total_norm_sq += grad.float().pow(2).sum()

    if is_fsdp and dist.is_initialized():
        dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
    return total_norm_sq.sqrt().item()


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Gradient clipping for FSDP with mixed-dtype gradients (fp32 + bf16).

    FSDP shards parameters across ranks, so each rank only holds a shard of
    gradients. We compute local norm in float32, all-reduce to get global norm,
    then clip.

    Returns:
        The global gradient L2 norm computed *before* clipping. Reuse this for
        logging instead of recomputing the (post-clip) norm separately.
    """
    

    is_fsdp = isinstance(model, FSDPModule)
    if not is_fsdp:
        # clip_grad_norm_ returns the total norm *before* clipping. Under DDP
        # grads are already all-reduced, so the local norm equals the global one.
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return float(total_norm)

    # Compute local sharded norm in float32 (handles mixed dtype)
    local_norm_sq = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            local_norm_sq += p.grad.detach().float().norm(2) ** 2

    # All-reduce to get global norm across all ranks
    if dist.is_initialized():
        torch.distributed.all_reduce(local_norm_sq)
    total_norm = local_norm_sq.sqrt()

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(clip_coef.to(p.grad.dtype))

    return total_norm.item()


def clean_nan_gradients(model: nn.Module):
    """Replace NaN/Inf gradients with 0."""
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)

