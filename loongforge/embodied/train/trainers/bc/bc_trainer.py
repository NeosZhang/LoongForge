# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""BCTrainer — Behavior Cloning training paradigm."""

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class BCTrainer(BaseTrainer):
    """
    Behavior Cloning Trainer.

    Simple BC: forward(batch) → action_loss → backward → step.
    Supports the standard ModelFramework forward interface.
    """

    def _build_model(self) -> nn.Module:
        from loongforge.embodied.model import build_model
        return build_model(self.model_cfg)

    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        from loongforge.embodied.data import build_dataloader

        dl = build_dataloader(self.model_cfg, self.args, self.ctx)
        return {"vla": dl}

    def _train_forward(self, batch) -> Dict[str, torch.Tensor]:
        """BC forward: call model(batch), expect 'action_loss' in output."""
        dtype = getattr(self, "_compute_dtype", None)
        if dtype is None:
            from loongforge.embodied.distributed.parallel import _resolve_dtype
            dtype = _resolve_dtype(self.args.dtype)
            self._compute_dtype = dtype

        if self.args.enable_autocast:
            with torch.autocast("cuda", dtype=dtype):
                return self.model(batch)
        else:
            return self.model(batch)

    def _on_train_begin(self):
        if self.ctx.is_main:
            from loongforge.embodied.distributed.parallel import unwrap_model

            model = unwrap_model(self.model)
            logger.info(f"Model: {model.__class__.__name__}")
