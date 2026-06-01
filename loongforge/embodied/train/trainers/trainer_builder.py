# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Trainer registration and construction — mirrors main framework pattern."""

import logging
from typing import Callable, List, Union

logger = logging.getLogger(__name__)

_TRAINER_REGISTRY = {}


def register_model_trainer(model_family: Union[str, List[str]], training_phase: str):
    """
    Decorator: register a trainer builder function for (model_family, training_phase).

    Usage:
        @register_model_trainer("pi05", "finetune")
        def pi05_finetune(args):
            return BCTrainer(args)
    """

    def decorator(fn: Callable):
        families = [model_family] if isinstance(model_family, str) else model_family
        for family in families:
            key = (family.lower(), training_phase.lower())
            if key in _TRAINER_REGISTRY:
                logger.warning(f"Overriding trainer for {key}")
            _TRAINER_REGISTRY[key] = fn
        return fn

    return decorator


def build_model_trainer(args):
    """Look up and build Trainer by model_cfg.model_type + args.training_phase."""
    # Ensure all trainer modules are imported (triggers @register decorators)
    _auto_import_trainers()

    model_type = args.model_cfg.model_type
    phase = args.training_phase
    key = (model_type.lower(), phase.lower())

    if key not in _TRAINER_REGISTRY:
        available = [f"{k[0]}:{k[1]}" for k in sorted(_TRAINER_REGISTRY.keys())]
        raise ValueError(
            f"No trainer registered for (model_type={model_type}, phase={phase}). "
            f"Available: {available}"
        )

    builder_fn = _TRAINER_REGISTRY[key]
    return builder_fn(args)


def _auto_import_trainers():
    """Auto-import trainer modules to trigger @register decorators."""
    # pylint: disable=unused-import
    import loongforge.embodied.train.trainers.bc.bc_trainer  # noqa: F401
