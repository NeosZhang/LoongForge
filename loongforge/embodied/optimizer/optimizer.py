# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer construction."""

import torch
import torch.nn as nn


OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Build optimizer with per-module LR groups; class selected via args.optimizer."""
    from embodied.optimizer.lr import build_param_groups

    groups = build_param_groups(model, args)
    optimizer_cls = OPTIMIZER_REGISTRY.get(args.optimizer)
    if optimizer_cls is None:
        supported = ", ".join(OPTIMIZER_REGISTRY)
        raise ValueError(f"Unknown optimizer '{args.optimizer}'. Supported optimizers: {supported}.")

    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if optimizer_cls in (torch.optim.AdamW, torch.optim.Adam):
        kwargs.update(betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)

    return optimizer_cls(groups, **kwargs)
