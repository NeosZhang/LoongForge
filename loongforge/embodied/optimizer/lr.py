# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-module LR groups + scheduler factory."""

import logging
from typing import Dict, List

import torch.nn as nn

from loongforge.embodied.distributed.utils import unwrap_model

logger = logging.getLogger(__name__)


def build_param_groups(model: nn.Module, args) -> List[Dict]:
    """
    Build optimizer param groups with per-module LR from CLI args.

    Supports:
      --lr-backbone → backbone / architecture.pi05_model.paligemma_with_expert
      --lr-action-model → action_model / architecture.pi05_model.action_expert
      --lr → everything else (base)
    """
    raw = unwrap_model(model)
    frozen_ids = {id(p) for p in raw.parameters() if not p.requires_grad}
    used_ids = set()
    groups = []

    # Define LR override mappings: try multiple module paths for each CLI arg
    lr_mappings = []
    if args.lr_backbone is not None:
        lr_mappings.append((args.lr_backbone, [
            "backbone",
            "architecture.backbone",
            "architecture.pi05_model.paligemma_with_expert",
        ]))
    if args.lr_action_model is not None:
        lr_mappings.append((args.lr_action_model, [
            "action_model",
            "architecture.action_head",
            "architecture.pi05_model.action_expert",
            "architecture.pi05_model",
        ]))

    for lr_val, candidate_paths in lr_mappings:
        for path in candidate_paths:
            module = raw
            try:
                for attr in path.split("."):
                    module = getattr(module, attr)
            except AttributeError:
                continue

            params = [
                p for p in module.parameters()
                if p.requires_grad and id(p) not in frozen_ids and id(p) not in used_ids
            ]
            if params:
                groups.append({"params": params, "lr": lr_val, "name": path})
                used_ids.update(id(p) for p in params)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"LR group '{path}': lr={lr_val}, params={len(params)}")
                break  # Only use first matching path

    # Base group: everything else
    other = [
        p for p in raw.parameters()
        if p.requires_grad and id(p) not in used_ids and id(p) not in frozen_ids
    ]
    if other:
        groups.append({"params": other, "lr": args.lr, "name": "base"})

    return groups


def build_scheduler(optimizer, args):
    """Build LR scheduler from CLI args."""
    from transformers import get_scheduler

    kwargs = {}
    if args.min_lr is not None:
        kwargs["min_lr"] = args.min_lr

    return get_scheduler(
        name=args.lr_decay_style,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_iters,
        num_training_steps=args.train_iters,
        scheduler_specific_kwargs=kwargs,
    )
