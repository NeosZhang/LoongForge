# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer construction."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Build optimizer with per-module LR groups; class selected via args.optimizer.

    If --zero-optimizer is set and strategy is DDP, wraps with
    ZeroRedundancyOptimizer to shard optimizer states across ranks.
    """
    from loongforge.embodied.optimizer.lr import build_param_groups

    groups = build_param_groups(model, args)
    optimizer_cls = OPTIMIZER_REGISTRY.get(args.optimizer)
    if optimizer_cls is None:
        supported = ", ".join(OPTIMIZER_REGISTRY)
        raise ValueError(f"Unknown optimizer '{args.optimizer}'. Supported optimizers: {supported}.")

    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if optimizer_cls in (torch.optim.AdamW, torch.optim.Adam):
        kwargs.update(betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)

    # ZeRO Stage-1: shard optimizer states across DDP ranks
    use_zero = getattr(args, "zero_optimizer", False)
    if use_zero:
        strategy = getattr(args, "distributed_strategy", "ddp")
        if strategy == "ddp":
            from torch.distributed.optim import ZeroRedundancyOptimizer
            logger.info("Using ZeroRedundancyOptimizer (ZeRO Stage-1) with DDP")
            return ZeroRedundancyOptimizer(
                groups,
                optimizer_class=optimizer_cls,
                parameters_as_bucket_view=True,
                **kwargs,
            )
        else:
            logger.warning(
                f"--zero-optimizer ignored: only effective with --distributed-strategy ddp, "
                f"current strategy is '{strategy}' (already shards optimizer states)."
            )

    return optimizer_cls(groups, **kwargs)

