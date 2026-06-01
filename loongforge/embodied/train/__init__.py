# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training control layer."""

from .parser import parse_train_args
from .global_vars import get_args, get_model_config
from .trainers.trainer_builder import build_model_trainer, register_model_trainer

__all__ = [
    "parse_train_args",
    "get_args",
    "get_model_config",
    "build_model_trainer",
    "register_model_trainer",
]
