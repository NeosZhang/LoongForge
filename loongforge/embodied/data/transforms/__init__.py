# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
LoongForge VLA Data Transforms & Collators

Per-sample transforms:
  - BaseTransform / ComposedTransform: Transform base classes
  - Normalizer: Multi-mode normalization (q99, min_max, mean_std, scale, binary)
  - ImageTransform: Image preprocessing (configurable resize strategy + normalize mode)
  - ActionTransform: Action chunking + normalization (configurable padding strategy)
  - StateDiscretizationTransform: State discretization for text-conditioned VLA models

Batch-level collators (DataLoader collate_fn):
  - BasePreprocessor / PreparedBatch: Base classes
  - Pi05Preprocessor / Pi05PreparedBatch: Pi0.5 collator
  - register_preprocessor: Registry decorator

Utilities:
  - tokenize_prompts, build_tokenizer
  - convert_stats: Convert dataset stats to numpy format
"""

from loongforge.embodied.data.transforms.base import BaseTransform, ComposedTransform
from loongforge.embodied.data.transforms.normalizer import Normalizer
from loongforge.embodied.data.transforms.image_transform import ImageTransform
from loongforge.embodied.data.transforms.action_transform import ActionTransform
from loongforge.embodied.data.transforms.pipeline import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
    build_transforms_from_args,
    convert_stats,
)
from loongforge.embodied.data.transforms.pi05_collator import (
    StateDiscretizationTransform,
    Pi05Preprocessor,
    Pi05PreparedBatch,
    tokenize_prompts,
    build_tokenizer,
)

__all__ = [
    # Per-sample transforms
    "BaseTransform",
    "ComposedTransform",
    "Normalizer",
    "ImageTransform",
    "ActionTransform",
    "StateDiscretizationTransform",
    "convert_stats",
    "build_transforms_from_args",
    # Collator framework
    "BasePreprocessor",
    "PreparedBatch",
    "register_preprocessor",
    # Pi05 collator
    "Pi05Preprocessor",
    "Pi05PreparedBatch",
    "tokenize_prompts",
    "build_tokenizer",
]
