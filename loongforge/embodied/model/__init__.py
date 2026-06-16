# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied Model - Unified model building entry point."""

import importlib
import pkgutil
from pathlib import Path

from loongforge.embodied.model.registry import MODEL_REGISTRY, register_model, build_model

# Auto-import all subpackages to trigger @register_model decorators
for _info in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if _info.ispkg:
        importlib.import_module(f"loongforge.embodied.model.{_info.name}")

__all__ = ["build_model", "register_model", "MODEL_REGISTRY"]
