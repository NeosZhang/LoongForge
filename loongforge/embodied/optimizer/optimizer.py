# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer construction."""

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Build AdamW with per-module LR groups."""
    from embodied.optimizer.lr import build_param_groups

    groups = build_param_groups(model, args)
    return torch.optim.AdamW(
        groups,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
