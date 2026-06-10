# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-module LR groups + scheduler factory."""

import logging
from typing import Dict, List

import torch.nn as nn

from loongforge.embodied.distributed.utils import is_rank_zero, unwrap_model

logger = logging.getLogger(__name__)


def _log_model_lr(model: nn.Module, max_depth: int = 3, groups: List[Dict] = None) -> None:
    """Log named submodules with trainable parameter counts, and optionally their LR assignment.

    When ``groups`` is provided, each module row also shows the lr value assigned to
    its parameters (aggregated from the first param found in that module).
    Only logs on rank 0 (or when distributed is not initialized).
    """

    if not is_rank_zero():
        return

    # Build param_id → lr mapping when groups are available
    param_to_lr: dict = {}
    if groups is not None:
        for group in groups:
            for p in group.get("params", []):
                param_to_lr[id(p)] = group["lr"]

    if groups is not None:
        title = "[LR Groups] Model modules with LR assignment:"
    else:
        title = (
            "[LR Groups] Model modules"
            " (use paths below with --lr-group to set per-module LR):"
        )
    lines = [title]
    for name, module in model.named_modules():
        if not name:
            continue
        depth = name.count(".")
        if depth >= max_depth:
            continue
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if trainable == 0:
            continue
        indent = "  " + "  " * depth
        if trainable >= 1_000_000:
            param_str = f"{trainable / 1e6:.1f}M"
        elif trainable >= 1_000:
            param_str = f"{trainable / 1e3:.1f}K"
        else:
            param_str = str(trainable)

        if groups is not None:
            lrs = {param_to_lr[id(p)] for p in module.parameters() if p.requires_grad and id(p) in param_to_lr}
            if not lrs:
                lr_str = "  lr=frozen"
            elif len(lrs) == 1:
                lr_str = f"  lr={lrs.pop():.2e}"
            else:
                lr_str = "  lr=mixed(" + ", ".join(f"{v:.2e}" for v in sorted(lrs)) + ")"
        else:
            lr_str = ""

        lines.append(f"{indent}{name:<60s}  ({param_str} trainable params){lr_str}")
    logger.info("\n".join(lines))


def _parse_lr_group(lr_group_str: str) -> list[tuple[str, float]]:
    """Parse 'path1=lr1,path2=lr2' into an ordered [(path, lr)] list."""
    result = []
    for item in lr_group_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --lr-group entry '{item}': expected 'module.path=lr'"
            )
        path, lr_str = item.rsplit("=", 1)
        result.append((path.strip(), float(lr_str.strip())))
    return result


def build_param_groups(model: nn.Module, training_args) -> List[Dict]:
    """Build optimizer param groups with per-module LR from CLI training_args.

    LR assignment priority (highest to lowest):
      1. ``--lr-group``  — comma-separated ``module.path=lr`` pairs.
         Entries are processed in order; earlier entries consume parameters
         first, so more specific (deeper) paths should be listed before
         broader ancestor paths.
         Example: ``model.paligemma_with_expert.gemma_expert=1e-4,
                   model.paligemma_with_expert=1e-5``
      2. ``--lr-base``  — fallback for all remaining trainable parameters.

    Parameters are never double-counted: once a parameter is assigned to a
    group it is excluded from all subsequent groups.
    """
    raw = unwrap_model(model)
    frozen_ids = {id(p) for p in raw.parameters() if not p.requires_grad}
    used_ids = set()
    groups = []

    base_lr = training_args.lr_base

    _log_model_lr(raw)

    lr_mappings: list[tuple[str, float]] = []
    lr_group_str = training_args.lr_group

    if lr_group_str:
        lr_mappings = _parse_lr_group(lr_group_str)

    for path, lr_val in lr_mappings:
        module = raw
        try:
            for attr in path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            continue

        parameters = module.parameters() if isinstance(module, nn.Module) else [module]
        params = [
            p for p in parameters
            if p.requires_grad and id(p) not in frozen_ids and id(p) not in used_ids
        ]
        if params:
            groups.append({"params": params, "lr": lr_val, "name": path})
            used_ids.update(id(p) for p in params)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"LR group '{path}': lr={lr_val}, params={len(params)}")

    # Base group: everything else
    other = [
        p for p in raw.parameters()
        if p.requires_grad and id(p) not in used_ids and id(p) not in frozen_ids
    ]
    if other:
        groups.append({"params": other, "lr": base_lr, "name": "base"})

    _log_model_lr(raw, 3, groups=groups)

    return groups


def build_scheduler(optimizer, training_args):
    """Build LR scheduler from CLI training_args."""

    if training_args.lr_decay_style == "lambda_linear":
        from torch.optim.lr_scheduler import LambdaLR
        from loongforge.embodied.optimizer.lr_scheduler import LambdaLinearScheduler

        f_max = getattr(training_args, "lambda_f_max", 0.4)
        f_min = getattr(training_args, "lambda_f_min", 0.0)
        f_start = getattr(training_args, "lambda_f_start", 0.0)
        cycle_len = getattr(training_args, "lambda_cycle_length", 10000) or training_args.train_iters

        _scheduler = LambdaLinearScheduler(
            warm_up_steps=[training_args.lr_warmup_iters],
            f_min=[f_min],
            f_max=[f_max],
            f_start=[f_start],
            cycle_lengths=[cycle_len]
        )

        logger.info(
            f"LambdaLinear scheduler: f_max={f_max}, f_min={f_min}, warmup={training_args.lr_warmup_iters}, "
            f"cycle_len={cycle_len}"
        )

        return LambdaLR(optimizer, _scheduler.schedule)

    from transformers import get_scheduler

    kwargs = {}
    if training_args.min_lr is not None:
        kwargs["min_lr"] = training_args.min_lr

    return get_scheduler(
        name=training_args.lr_decay_style,
        optimizer=optimizer,
        num_warmup_steps=training_args.lr_warmup_iters,
        num_training_steps=training_args.train_iters,
        scheduler_specific_kwargs=kwargs,
    )
