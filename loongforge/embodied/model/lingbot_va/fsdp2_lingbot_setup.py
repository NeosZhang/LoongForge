# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot FSDP2 post-wrap and optimizer-owned gradient helpers."""

from collections import defaultdict

import torch
from torch.distributed.tensor import DTensor


_LINGBOT_FSDP2_SETUP_DONE = "_lingbot_fsdp2_setup_done"
_LINGBOT_DTENSOR_CLIP_LOGGED = False


def _lingbot_optimizer_parameters(optimizer):
    """Return each trainable optimizer-owned parameter exactly once."""
    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen or not parameter.requires_grad:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
    return parameters


def _lingbot_local_gradient_groups(optimizer):
    """Collect mutable local DTensor gradients by device and dtype."""
    parameters = _lingbot_optimizer_parameters(optimizer)
    non_dtensor = [
        parameter for parameter in parameters if not isinstance(parameter, DTensor)
    ]
    if non_dtensor:
        raise RuntimeError(
            "LingBot optimizer-owned gradient handling requires pure DTensor parameters; "
            f"found {len(non_dtensor)} non-DTensor parameters."
        )

    groups = defaultdict(list)
    gradient_count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not isinstance(gradient, DTensor):
            raise RuntimeError(
                "LingBot optimizer-owned gradient handling requires DTensor gradients; "
                f"got {type(gradient).__name__}."
            )
        local_gradient = gradient._local_tensor
        if local_gradient.is_sparse:
            raise RuntimeError(
                "LingBot FSDP2 gradient handling does not support sparse gradients."
            )
        groups[(local_gradient.device, local_gradient.dtype)].append(local_gradient)
        gradient_count += 1
    return parameters, list(groups.values()), gradient_count


def _lingbot_local_norm_sq(gradient_groups, device):
    total_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
    for gradients in gradient_groups:
        norms = torch._foreach_norm(gradients, 2.0, dtype=torch.float32)
        total_norm_sq += torch.stack(norms).square().sum()
    return total_norm_sq


def clip_lingbot_optimizer_gradients(optimizer, max_norm):
    """Clip RAB=false optimizer-owned DTensor gradients by global L2 norm.

    FSDP2 leaves the reduced sharded gradients on the DTensor parameters held
    by the optimizer.  The model's currently materialized parameters may have
    no ``.grad``, so this helper intentionally starts from ``param_groups``.
    """
    global _LINGBOT_DTENSOR_CLIP_LOGGED

    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}.")
    parameters, gradient_groups, gradient_count = _lingbot_local_gradient_groups(
        optimizer
    )
    if parameters:
        device = parameters[0]._local_tensor.device
    else:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    total_norm_sq = _lingbot_local_norm_sq(gradient_groups, device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_norm_sq, op=torch.distributed.ReduceOp.SUM)
    total_norm = total_norm_sq.sqrt()
    clip_coefficient = torch.clamp(
        torch.as_tensor(max_norm, device=device, dtype=torch.float32)
        / (total_norm + 1e-6),
        max=1.0,
    )
    for gradients in gradient_groups:
        torch._foreach_mul_(gradients, clip_coefficient)

    if not _LINGBOT_DTENSOR_CLIP_LOGGED and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        print(
            "[lingbot-dtensor-clip] "
            f"optimizer_params={len(parameters)} dtensor_params={len(parameters)} "
            f"gradients={gradient_count} global_grad_norm={total_norm.item():.10g} "
            f"clip_coefficient={clip_coefficient.item():.10g}",
            flush=True,
        )
        _LINGBOT_DTENSOR_CLIP_LOGGED = True
    return total_norm.item()


def clean_lingbot_optimizer_gradients(optimizer):
    """Replace NaN/Inf in optimizer-owned local DTensor gradients with zero."""
    _, gradient_groups, _ = _lingbot_local_gradient_groups(optimizer)
    for gradients in gradient_groups:
        for gradient in gradients:
            torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0, out=gradient)


def register_lingbot_post_step_reshard(model, optimizer):
    """Register LingBot's post-optimizer FSDP2 reshard policy.

    The returned handle must be kept alive by the trainer. The optimizer's
    standard post-step hook guarantees that reshard runs after AdamW and
    before the public scheduler advances.
    """
    fsdp_modules = []
    seen = set()
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if id(module) in seen or not (
                hasattr(module, "unshard") and hasattr(module, "reshard")
            ):
                continue
            seen.add(id(module))
            fsdp_modules.append(module)

    logged = False

    def post_step_reshard(_optimizer, _args, _kwargs):
        nonlocal logged
        for module in fsdp_modules:
            module.reshard()
        if not logged and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            print(
                f"[lingbot-post-step-reshard] active modules={len(fsdp_modules)}",
                flush=True,
            )
            logged = True

    if not hasattr(optimizer, "register_step_post_hook"):
        raise TypeError(
            "LingBot optimizer must expose register_step_post_hook for post-step reshard"
        )
    return optimizer.register_step_post_hook(post_step_reshard), len(fsdp_modules)


def apply_lingbot_fsdp2_tuning(model):
    """Keep FSDP2 parameters unsharded after backward (final RAB=false stack)."""
    if getattr(model, _LINGBOT_FSDP2_SETUP_DONE, False):
        return
    setattr(model, _LINGBOT_FSDP2_SETUP_DONE, True)

    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return

    modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
    for module in modules:
        if hasattr(module, "set_reshard_after_backward"):
            module.set_reshard_after_backward(False, recurse=False)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if modules and rank == 0:
        print(
            f"[lingbot-fsdp2] reshard_after_backward=False applied to {len(modules)} modules",
            flush=True,
        )
