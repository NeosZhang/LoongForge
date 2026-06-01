# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
framework Layers - Four-layer composable abstractions

Layer 1: Trainer      (training/trainers/)          - Training paradigm (Template Method)
Layer 2: Architecture (model/compose/architecture/) - Network structure (Abstract Factory)
Layer 3: Condition    (model/compose/condition/)    - Condition injection (Strategy)
Layer 4: Action       (model/compose/action/)       - Action  (Strategy)
"""

from loongforge.embodied.model.compose.registry import (
    ARCHITECTURE_REGISTRY,
    CONDITION_REGISTRY,
    ACTION_REGISTRY,
    TRAINER_REGISTRY,
)
from loongforge.embodied.model.compose.builder import ModelFrameworkBuilder, build_framework
from loongforge.embodied.model.compose.base import ModelFramework

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "CONDITION_REGISTRY",
    "ACTION_REGISTRY",
    "TRAINER_REGISTRY",
    "ModelFrameworkBuilder",
    "build_framework",
    "ModelFramework",
]
