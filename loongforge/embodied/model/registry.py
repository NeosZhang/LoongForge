# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Global model registry — maps model_type strings to model classes.

Usage:
    # Register
    @register_model("pi05")
    class Pi05Model(nn.Module): ...

    # Build
    model = build_model(cfg)   # cfg.model_type = "pi05"
"""

import importlib
import pkgutil
from typing import Dict, Type

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str):
    """Decorator that registers a model class into MODEL_REGISTRY."""
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator


def _auto_import_model_modules():
    """Auto-import all sub-packages under model/ to trigger @register_model decorators."""
    import loongforge.embodied.model as _model_pkg
    import os

    model_dir = os.path.dirname(_model_pkg.__file__)
    for _, pkg_name, is_pkg in pkgutil.iter_modules([model_dir]):
        if is_pkg and pkg_name not in ("compose", "modules", "__pycache__"):
            try:
                sub_pkg = importlib.import_module(f"loongforge.embodied.model.{pkg_name}")
                # import modeling_<pkg_name>.py to trigger registration
                modeling_mod = f"loongforge.embodied.model.{pkg_name}.modeling_{pkg_name}"
                importlib.import_module(modeling_mod)
            except ModuleNotFoundError:
                pass


def build_model(cfg) -> nn.Module:
    """Build a model instance by cfg.model_type.

    Args:
        cfg: OmegaConf / dict, must contain the model_type field.

    Returns:
        Initialized nn.Module.
    """
    _auto_import_model_modules()

    model_type = cfg.get("model_type") if hasattr(cfg, "get") else getattr(cfg, "model_type", None)
    if model_type is None:
        raise ValueError("cfg.model_type is required for build_model()")

    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type: '{model_type}'. "
            f"Registered: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_type]
    return cls.from_pretrained(cfg)
