# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Registry for model-specific per-sample transform builders."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

from loongforge.embodied.data.transforms.base import BaseTransform


@dataclass(frozen=True)
class TransformBuilderContext:
    """Shared inputs available to model-specific transform builders."""

    model_cfg: Any
    data_cfg: Any
    training_args: Any
    dataset: Any
    dataset_stats: dict[str, Any] | None


TransformBuilder = Callable[[TransformBuilderContext], Iterable[BaseTransform]]

_TRANSFORM_BUILDER_REGISTRY: Dict[str, TransformBuilder] = {}
_DISCOVERED_TRANSFORM_BUILDERS = False


def register_transform_builder(model_type: str):
    """Decorator to register a model-specific per-sample transform builder."""
    def decorator(builder: TransformBuilder) -> TransformBuilder:
        _TRANSFORM_BUILDER_REGISTRY[model_type] = builder
        return builder

    return decorator


def get_transform_builder(model_type: str) -> TransformBuilder:
    """Look up a registered transform builder by model type."""
    discover_transform_builders()
    if model_type not in _TRANSFORM_BUILDER_REGISTRY:
        raise ValueError(
            f"Unknown transform builder for model_type '{model_type}'. "
            f"Available: {sorted(_TRANSFORM_BUILDER_REGISTRY.keys())}"
        )
    return _TRANSFORM_BUILDER_REGISTRY[model_type]


def discover_transform_builders() -> None:
    """Import transform subpackages so decorator registration can run."""
    global _DISCOVERED_TRANSFORM_BUILDERS
    if _DISCOVERED_TRANSFORM_BUILDERS:
        return

    _DISCOVERED_TRANSFORM_BUILDERS = True
    package_dir = Path(__file__).resolve().parent
    package_name = __package__
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.ispkg and not module_info.name.startswith("_"):
            importlib.import_module(f"{package_name}.{module_info.name}")
