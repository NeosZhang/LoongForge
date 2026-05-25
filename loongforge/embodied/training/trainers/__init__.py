"""
Layer 1: Trainers - Training paradigm registration and construction

Three-level inheritance hierarchy:
    BaseTrainer (training loop skeleton)
    ├── BCBaseTrainer (behavior cloning paradigm)
    │   ├── BCPretrainTrainer      # Full-parameter pretraining
    │   ├── BCPosttrainTrainer     # Freeze backbone then train
    │   ├── BCFinetuneTrainer      # Partial freeze + LoRA fine-tuning
    │   ├── BCCoTrainTrainer       # Action + VLM joint training
    │   ├── BCContinueLearningTrainer  # Replay to prevent forgetting
    │   └── BCAuxiliaryTrainer     # Auxiliary objectives (video pred, STDP)
    └── RLBaseTrainer (reinforcement learning paradigm)

Usage:
    from training.trainers import build_trainer, TRAINER_REGISTRY
    trainer = build_trainer(cfg, model, accelerator, optimizer, lr_scheduler)
"""

from model.compose.registry import TRAINER_REGISTRY
from training.trainers.base_trainer import BaseTrainer
from training.trainers.bc import (
    BCBaseTrainer,
    BCTrainer,
)


def build_trainer(cfg, model, accelerator, optimizer, lr_scheduler, dataloaders) -> BaseTrainer:
    """Build the corresponding Trainer instance based on the trainer name in config."""
    trainer_name = cfg.framework.layers.trainer
    trainer_cls = TRAINER_REGISTRY[trainer_name]
    return trainer_cls(
        cfg=cfg,
        model=model,
        accelerator=accelerator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloaders=dataloaders,
    )


__all__ = [
    # Base classes
    "BaseTrainer",
    "BCBaseTrainer",
    # BC trainers
    "BCTrainer",
    # RL trainers
    # Backward compatibility
    "BCTrainer",
    # Factory
    "build_trainer",
    "TRAINER_REGISTRY",
]
