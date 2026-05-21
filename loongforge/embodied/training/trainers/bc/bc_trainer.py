"""
BCTrainer - BC fine-tuning trainer

Partial freezing + optional LoRA, suitable for task-specific fine-tuning scenarios.
"""

from typing import Dict, Any
import logging

import torch
from model.compose.registry import TRAINER_REGISTRY
from training.trainers.bc.bc_base_trainer import BCBaseTrainer

logger = logging.getLogger(__name__)


@TRAINER_REGISTRY.register("BCTrainer")
class BCTrainer(BCBaseTrainer):
    """
    BC fine-tuning trainer.

    Features:
      - Partial freezing: freeze backbone lower layers, unfreeze top-K layers
      - Optional LoRA/Adapter lightweight fine-tuning
      - Task-specific small datasets
      - Lower learning rate

    Config:
      trainer.finetune:
        use_lora: false
        lora_rank: 8
        lora_alpha: 16
        unfreeze_top_k: 2    # Unfreeze top K layers of backbone
    """

    def __init__(self, cfg, model, accelerator, optimizer, lr_scheduler, dataloaders):
        super().__init__(cfg, model, accelerator, optimizer, lr_scheduler, dataloaders)
        self.finetune_cfg = cfg.trainer.get("finetune", {})
        self.use_lora = self.finetune_cfg.get("use_lora", False)
        self.unfreeze_top_k = self.finetune_cfg.get("unfreeze_top_k", 0)

    def _get_freeze_config(self) -> Dict[str, bool]:
        """Fine-tuning mode: freeze backbone (but top-K layers are unfrozen in on_train_begin)."""
        cfg_freeze = self.cfg.trainer.get("freeze", {})
        return {
            "backbone": cfg_freeze.get("backbone", True),
            "condition": cfg_freeze.get("condition", False),
            "action_head": cfg_freeze.get("action_head", False),
        }

    def on_train_begin(self):
        """Apply freeze strategy first, then unfreeze top-K backbone layers, optionally apply LoRA."""
        super().on_train_begin()

        # Unfreeze top-K backbone layers
        if self.unfreeze_top_k > 0:
            self._unfreeze_top_k_layers()

        # Apply LoRA (if enabled)
        if self.use_lora:
            self._apply_lora()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Finetune: trainable params {trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)"
        )

    def _unfreeze_top_k_layers(self):
        """Unfreeze top-K layers of backbone."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return

        # Try to get backbone's compose list
        compose = None
        for attr in ("compose", "model.compose", "encoder.layer"):
            parts = attr.split(".")
            obj = backbone
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                compose = obj
                break

        if compose is None:
            logger.warning("Cannot find backbone layers for top-K unfreezing")
            return

        # Unfreeze last K layers
        for layer in compose[-self.unfreeze_top_k:]:
            for param in layer.parameters():
                param.requires_grad = True

        logger.info(f"Unfroze top {self.unfreeze_top_k} backbone layers")

    def _apply_lora(self):
        """
        Apply LoRA adapters.

        Supports two configuration modes:
          1. Simple mode: trainer.finetune.use_lora / lora_rank / lora_alpha
          2. Full mode: cfg.lora.enabled / rank / alpha / target_modules / ...

        Simple mode is converted to equivalent full mode configuration.
        """
        from training.trainer_utils.peft import is_lora_enabled, apply_lora
        from training.trainer_utils.peft.config import LoRASpec

        # Check if using full lora: config block
        if is_lora_enabled(self.cfg):
            self.model = apply_lora(self.model, self.cfg, print_summary=True)
            return

        # Simple mode: build LoRA from trainer.finetune
        lora_rank = self.finetune_cfg.get("lora_rank", 8)
        lora_alpha = self.finetune_cfg.get("lora_alpha", 16)
        lora_dropout = self.finetune_cfg.get("lora_dropout", 0.05)
        target_modules = self.finetune_cfg.get("lora_target_modules", "all-linear")
        vlm_module = self.finetune_cfg.get("lora_vlm_module", None)

        logger.info(f"LoRA enabled: rank={lora_rank}, alpha={lora_alpha}")

        try:
            from peft import get_peft_model, LoraConfig
            from training.trainer_utils.peft.injector import _detect_vlm_interface

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                init_lora_weights="gaussian",
            )

            # Resolve VLM interface
            if vlm_module:
                vlm_interface = getattr(self.model, vlm_module, None)
                if vlm_interface is None and hasattr(self.model, "architecture"):
                    vlm_interface = getattr(self.model.architecture, vlm_module, None)
            else:
                vlm_interface = _detect_vlm_interface(self.model)

            if vlm_interface is None:
                logger.warning(
                    "No VLM interface found for LoRA, applying to full model"
                )
                self.model = get_peft_model(self.model, lora_config)
            else:
                # Freeze VLM, inject LoRA
                for p in vlm_interface.parameters():
                    p.requires_grad = False
                if hasattr(vlm_interface, "model") and isinstance(
                    vlm_interface.model, torch.nn.Module
                ):
                    vlm_interface.model = get_peft_model(
                        vlm_interface.model, lora_config
                    )
                else:
                    from peft import get_peft_model
                    # Wrap the interface itself
                    get_peft_model(vlm_interface, lora_config)

            # Log trainable summary
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"LoRA applied: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M trainable "
                f"({100 * trainable / max(total, 1):.2f}%)"
            )

        except ImportError:
            logger.error(
                "peft library not installed. Install with: pip install peft"
            )
            raise
