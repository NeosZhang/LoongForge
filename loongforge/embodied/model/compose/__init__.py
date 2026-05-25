"""
framework Layers - Four-layer composable abstractions

Layer 1: Trainer      (training/trainers/)          - Training paradigm (Template Method)
Layer 2: Architecture (model/compose/architecture/) - Network structure (Abstract Factory)
Layer 3: Condition    (model/compose/condition/)    - Condition injection (Strategy)
Layer 4: Action       (model/compose/action/)       - Action  (Strategy)
"""

from model.compose.registry import (
    ARCHITECTURE_REGISTRY,
    CONDITION_REGISTRY,
    ACTION_REGISTRY,
    TRAINER_REGISTRY,
)
from model.compose.builder import ModelFrameworkBuilder, build_framework
from model.compose.base import ModelFramework

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "CONDITION_REGISTRY",
    "ACTION_REGISTRY",
    "TRAINER_REGISTRY",
    "ModelFrameworkBuilder",
    "build_framework",
    "ModelFramework",
]
