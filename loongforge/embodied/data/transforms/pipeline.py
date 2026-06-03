# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLA transforms pipeline: collator framework + pipeline builder.

Collator framework (DataLoader collate_fn base classes):
    - PreparedBatch: Base dataclass for model-ready batch tensors
    - BasePreprocessor: Abstract base for collate functions
    - register_preprocessor: Decorator to register model-specific collators

Pipeline builder:
    - convert_stats: Convert dataset statistics to numpy format
    - build_transforms_from_args: Build per-sample transforms from config + CLI args
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch

from loongforge.embodied.data.transforms.base import BaseTransform, ComposedTransform
from loongforge.embodied.data.transforms.image_transform import ImageTransform
from loongforge.embodied.data.transforms.action_transform import ActionTransform


# ═══════════════════════════════════════════════════════════════
# Preprocessor Registry
# ═══════════════════════════════════════════════════════════════

_PREPROCESSOR_REGISTRY: Dict[str, Type["BasePreprocessor"]] = {}


def register_preprocessor(name: str):
    """Decorator to register a preprocessor class for a model."""
    def decorator(cls):
        _PREPROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


# ═══════════════════════════════════════════════════════════════
# Base Classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class PreparedBatch:
    """Base class for preprocessed batch data.

    All tensor fields are on CPU after collation.
    Call .to(device) to move everything to GPU before forward().
    """
    def to(self, device: torch.device) -> "PreparedBatch":
        """Move all tensor fields to the given device. Returns self."""
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(device))
            elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                setattr(self, f.name, [t.to(device) for t in val])
        return self


class BasePreprocessor:
    """Abstract base for model-specific DataLoader collate functions."""

    @classmethod
    def from_config(cls, cfg) -> "BasePreprocessor":
        """Construct preprocessor from a full config object."""
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config(cfg) classmethod"
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> PreparedBatch:
        """Transform a list of dataset samples into a PreparedBatch."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
# Pipeline Builder
# ═══════════════════════════════════════════════════════════════

def convert_stats(stats_raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    """Convert dataset stats (torch.Tensor/list) to numpy for Normalizer."""
    if stats_raw is None:
        return None

    stats = {}
    for k, v in stats_raw.items():
        if isinstance(v, torch.Tensor):
            stats[k] = v.cpu().numpy()
        elif isinstance(v, np.ndarray):
            stats[k] = v
        elif isinstance(v, (list, tuple)):
            stats[k] = np.array(v)
        else:
            stats[k] = v
    return stats


def build_transforms_from_args(
    model_cfg,
    args,
    dataset,
    dataset_stats,
    extra_transforms: Optional[List[BaseTransform]] = None,
) -> Optional[ComposedTransform]:
    """Build per-sample transforms from flat model_cfg + CLI args.

    Args:
        model_cfg: Flat model configuration dict (backbone, action_model at top level)
        args: CLI args namespace
        dataset: The dataset instance (used to discover image keys)
        dataset_stats: Dict of normalization statistics from dataset.meta.stats
        extra_transforms: Additional model-specific transforms to append

    Returns:
        ComposedTransform or None if not applicable.
    """
    backbone_cfg = model_cfg.get("backbone", {}) if hasattr(model_cfg, "get") else {}
    action_cfg = model_cfg.get("action_model", {}) if hasattr(model_cfg, "get") else {}

    image_size = getattr(args, "image_size", backbone_cfg.get("image_size", 224))
    action_horizon = getattr(args, "action_horizon", action_cfg.get("action_horizon", 50))
    max_action_dim = action_cfg.get("max_action_dim", 32)
    normalization_mode = getattr(args, "normalization_mode", "q99")

    # Discover image keys from first sample
    try:
        first_sample = dataset[0]
        image_keys = sorted(k for k in first_sample.keys() if k.startswith("observation.images."))
    except Exception:
        image_keys = []

    transforms = []

    # 1. Image transform (common to all VLA models)
    if image_keys:
        transforms.append(ImageTransform(
            apply_to=image_keys,
            image_size=image_size,
        ))

    # 2. Action transform (common to all VLA models)
    action_stats = convert_stats(dataset_stats.get("action")) if dataset_stats else None
    transforms.append(ActionTransform(
        apply_to=["action"],
        action_horizon=action_horizon,
        max_action_dim=max_action_dim,
        normalization_mode=normalization_mode,
        statistics=action_stats,
    ))

    # 3. Append model-specific transforms
    if extra_transforms:
        transforms.extend(extra_transforms)

    if not transforms:
        return None

    return ComposedTransform(transforms)
