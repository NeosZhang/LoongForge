"""
BCBaseTrainer - Behavior cloning paradigm intermediate base class

Provides common logic for standard BC training:
  - forward/backward/clip/step template
  - Parameter freezing strategy framework
  - Subclasses customize behavior by overriding _bc_forward / _compute_total_loss / _get_freeze_config
"""

from typing import Dict, Any
import logging

import torch
from torch.utils.data import DataLoader

from training.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class BCBaseTrainer(BaseTrainer):
    """
    Behavior cloning paradigm intermediate base class.

    train_step provides a fixed BC training template:
        zero_grad → _bc_forward → _compute_total_loss → _backward_and_step

    Subclasses customize behavior by overriding the following methods:
        - _bc_forward(batches): Forward pass logic
        - _compute_total_loss(output): Loss computation logic
        - _get_freeze_config(): Parameter freezing strategy
    """

    def __init__(self, cfg, model, accelerator, optimizer, lr_scheduler, dataloaders):
        super().__init__(cfg, model, accelerator, optimizer, lr_scheduler)
        self._external_dataloaders: Dict[str, DataLoader] = dataloaders

    def prepare_training(self):
        """Parse BC-specific CLI args and merge into cfg, then call base prepare_training."""
        from training.trainers.bc.bc_args import parse_bc_args, merge_bc_args_to_cfg
        bc_args = parse_bc_args()
        self.cfg = merge_bc_args_to_cfg(bc_args, self.cfg)
        super().prepare_training()

    # ═══════════════════════════════════════════════════════════════
    # BaseTrainer abstract method implementations
    # ═══════════════════════════════════════════════════════════════

    def setup_dataloaders(self) -> Dict[str, DataLoader]:
        """Default config: single VLA data loader. Subclasses can override to add more."""
        return {"vla": self._external_dataloaders.get("vla")}

    def train_step(self, batches: Dict[str, Any]) -> Dict[str, float]:
        """
        BC training single step template:
          1. zero_grad
          2. _bc_forward → output
          3. _compute_total_loss → total_loss
          4. _backward_and_step
          5. _collect_metrics
        """
        self.optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self._bc_forward(batches)
            total_loss = self._compute_total_loss(output)

        self._backward_and_step(total_loss)
        return self._collect_metrics(output, total_loss)

    # ═══════════════════════════════════════════════════════════════
    # Overridable BC template methods
    # ═══════════════════════════════════════════════════════════════

    def _bc_forward(self, batches: Dict[str, Any]) -> Dict[str, Any]:
        """
        BC forward pass. Default: single-path VLA forward.

        Subclasses override this method for multi-path forward (e.g., co-train).

        Returns:
            output dict, must contain at least "action_loss" key
        """
        return self.model.forward(batches["vla"])

    def _compute_total_loss(self, output: Dict[str, Any]) -> torch.Tensor:
        """
        Compute total loss. Default: directly use action_loss or total_loss.

        Subclasses override this method for multi-objective weighted combination.
        """
        return output.get("total_loss", output["action_loss"])

    def _get_freeze_config(self) -> Dict[str, bool]:
        """
        Return parameter freezing configuration.

        Returns:
            dict, key is module name (backbone, condition, action_head),
            value is whether to freeze. Default: all unfrozen.
        """
        return self.cfg.trainer.get("freeze", {})

    # ═══════════════════════════════════════════════════════════════
    # Hook implementations
    # ═══════════════════════════════════════════════════════════════

    def on_train_begin(self):
        """Freeze corresponding parameters based on freeze config."""
        freeze_cfg = self._get_freeze_config()
        if not freeze_cfg:
            return

        for module_name, should_freeze in freeze_cfg.items():
            if not should_freeze:
                continue
            module = getattr(self.model, module_name, None)
            if module is None:
                logger.warning(f"Freeze config: module '{module_name}' not found, skipping")
                continue
            for param in module.parameters():
                param.requires_grad = False
            logger.info(f"Frozen module: {module_name}")

    # ═══════════════════════════════════════════════════════════════
    # Internal utility methods
    # ═══════════════════════════════════════════════════════════════

    def _backward_and_step(self, total_loss: torch.Tensor):
        """Common backward pass + loss spike protection + NaN gradient cleanup + gradient clipping + optimizer step."""
        # Loss spike protection
        total_loss = self._check_loss_spike(total_loss)

        self.accelerator.backward(total_loss)

        # NaN gradient cleanup + gradient clipping
        if self.gradient_clipping > 0:
            self._clean_nan_gradients()
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clipping,
            )

        self.optimizer.step()
        self.lr_scheduler.step()

    def _collect_metrics(self, output: Dict[str, Any], total_loss: torch.Tensor) -> Dict[str, float]:
        """Collect training metrics."""
        metrics = {
            "total_loss": total_loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }
        # Collect all loss-related scalars from output
        for key, value in output.items():
            if "loss" in key and isinstance(value, torch.Tensor):
                metrics[key] = value.item()
        return metrics

    def set_dataloaders(self, dataloaders: Dict[str, DataLoader]):
        """Externally inject dataloaders."""
        self._external_dataloaders = dataloaders
