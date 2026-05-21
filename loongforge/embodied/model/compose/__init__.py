"""
AlphaFramework Layers - Four-layer composable abstractions

Layer 1: Trainer      (training/trainers/)    - Training paradigm (Template Method)
Layer 2: Architecture (model/compose/architecture/) - Network structure (Abstract Factory)
Layer 3: Condition    (model/compose/condition/)    - Condition injection (Strategy)
Layer 4: ActionLoss   (model/compose/action_loss/)  - Action loss (Strategy)
"""

from model.compose.registry import (
    ARCHITECTURE_REGISTRY,
    CONDITION_REGISTRY,
    LOSS_REGISTRY,
    TRAINER_REGISTRY,
)
from model.compose.builder import LayeredFrameworkBuilder, build_framework
from model.compose.base import LayeredFramework

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "CONDITION_REGISTRY",
    "LOSS_REGISTRY",
    "TRAINER_REGISTRY",
    "LayeredFrameworkBuilder",
    "build_framework",
    "LayeredFramework",
]
