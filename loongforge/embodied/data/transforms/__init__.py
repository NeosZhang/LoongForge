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
  - tokenize_prompts
  - convert_stats: Convert dataset stats to numpy format
"""

from loongforge.embodied.data.transforms.base import BaseTransform, ComposedTransform
from loongforge.embodied.data.transforms.normalizer import Normalizer
from loongforge.embodied.data.transforms.image_transform import ImageTransform
from loongforge.embodied.data.transforms.action_transform import ActionTransform
from loongforge.embodied.data.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
    get_preprocessor,
    build_preprocessor,
)
from loongforge.embodied.data.transforms.pipeline import (
    build_transforms_from_args,
    convert_stats,
)
from loongforge.embodied.data.transforms.pi05 import (
    StateDiscretizationTransform,
    Pi05Preprocessor,
    Pi05PreparedBatch,
    Pi05CollateImagesTransform,
    Pi05FallbackPromptTransform,
    Pi05TokenizeTransform,
    tokenize_prompts,
)
from loongforge.embodied.data.transforms.groot_n1_6 import (
    GrootBatchTransform,
    GrootN1d6FeatureTransform,
    GrootN1d6PreparedBatch,
    GrootN1d6Preprocessor,
    GrootPromptTransform,
    GrootStateActionTransform,
)

__all__ = [
    # Per-sample transforms
    "BaseTransform",
    "ComposedTransform",
    "Normalizer",
    "ImageTransform",
    "ActionTransform",
    "StateDiscretizationTransform",
    "Pi05CollateImagesTransform",
    "Pi05FallbackPromptTransform",
    "Pi05TokenizeTransform",
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
    # GR00T-N1.6 collator
    "GrootBatchTransform",
    "GrootN1d6FeatureTransform",
    "GrootPromptTransform",
    "GrootStateActionTransform",
    "GrootN1d6Preprocessor",
    "GrootN1d6PreparedBatch",
]
