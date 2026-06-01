# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""BCTrainer — Behavior Cloning training paradigm."""

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base_trainer import BaseTrainer
from ..trainer_builder import register_model_trainer

logger = logging.getLogger(__name__)


@register_model_trainer(["pi05", "groot_n1_6"], "finetune")
@register_model_trainer(["pi05", "groot_n1_6"], "pretrain")
def _build_bc_trainer(args):
    return BCTrainer(args)


class BCTrainer(BaseTrainer):
    """
    Behavior Cloning Trainer.

    Simple BC: forward(batch) → action_loss → backward → step.
    Supports the standard ModelFramework forward interface.
    """

    def _build_model(self) -> nn.Module:
        from embodied.model import build_model

        return build_model(self.model_cfg)

    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        from embodied.data import build_dataloader

        dl = build_dataloader(self.model_cfg, self.args, self.ctx)
        return {"vla": dl}

    def _train_forward(self, batch) -> Dict[str, torch.Tensor]:
        """BC forward: call model(batch), expect 'action_loss' in output."""
        dtype = getattr(self, "_compute_dtype", None)
        if dtype is None:
            from embodied.distributed.parallel import _resolve_dtype
            dtype = _resolve_dtype(self.args.dtype)
            self._compute_dtype = dtype

        with torch.autocast("cuda", dtype=dtype):
            return self.model(batch)

    def _on_train_begin(self):
        if self.ctx.is_main:
            from embodied.distributed.parallel import unwrap_model

            model = unwrap_model(self.model)
            arch = getattr(model, "architecture", None)
            if arch:
                logger.info(f"Architecture: {arch.__class__.__name__}")
