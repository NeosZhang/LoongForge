# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLA transforms pipeline builder.

Pipeline builder:
    - convert_stats: Convert dataset statistics to numpy format
    - build_transforms_from_args: Build per-sample transforms from config + CLI args
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from loongforge.embodied.data.transforms.base import BaseTransform, ComposedTransform
from loongforge.embodied.data.transforms.image_transform import ImageTransform
from loongforge.embodied.data.transforms.action_transform import ActionTransform


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
) -> Optional[ComposedTransform]:
    """Build per-sample transforms from flat model_cfg + CLI args.

    Args:
        model_cfg: Flat model configuration dict (backbone, action_model at top level)
        args: CLI args namespace
        dataset: The dataset instance (used to discover image keys)
        dataset_stats: Dict of normalization statistics from dataset.meta.stats

    Returns:
        ComposedTransform or None if not applicable.
    """


    image_size = (getattr(args, "image_size", None)
                  or model_cfg.get("image_size", 224))
    action_horizon = (getattr(args, "action_horizon", None)
                      or model_cfg.get("action_horizon", None))
    max_action_dim = (model_cfg.get("max_action_dim", None))
    normalization_mode = getattr(args, "normalization_mode", "q99")

    # Discover image keys from first sample
    try:
        first_sample = dataset[0]
        image_keys = sorted(k for k in first_sample.keys() if k.startswith("observation.images."))
    except Exception:
        image_keys = []

    transforms = []

    # 1. Image transform (configurable via backbone config)
    if image_keys:
        img_normalize_mode = model_cfg.get("image_normalize_mode", "identity")
        img_resize_strategy = model_cfg.get("image_resize_strategy", "resize_with_pad")

        transforms.append(ImageTransform(
            apply_to=image_keys,
            image_size=image_size,
            resize_strategy=img_resize_strategy,
            normalize_mode=img_normalize_mode,
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

    # 3. Pi05-specific transforms: state discretization, collate images, fallback prompt, tokenize
    _append_pi05_transforms(transforms, model_cfg, model_cfg, args, dataset_stats, image_size)

    # 4. Fast-specific transforms: key mapping (images→PIL, action→numpy, task→lang)
    _append_fast_transforms(transforms, model_cfg, image_size)

    if not transforms:
        return None

    return ComposedTransform(transforms)


def _append_pi05_transforms(transforms, model_cfg, backbone_cfg, args, dataset_stats, image_size):
    """Append Pi05-specific per-sample transforms if model_type is pi05."""
    model_type = model_cfg.get("model_type", "") if hasattr(model_cfg, "get") else ""
    if model_type != "pi05":
        return

    import os
    from loongforge.embodied.data.transforms.pi05.pi05_transform import (
        StateDiscretizationTransform,
        Pi05CollateImagesTransform,
        Pi05FallbackPromptTransform,
        Pi05TokenizeTransform,
    )

    normalization_mode = getattr(args, "normalization_mode", "q99")
    state_stats = convert_stats(dataset_stats.get("observation.state")) if dataset_stats else None

    transforms.append(StateDiscretizationTransform(
        apply_to=["prompt"],
        state_key="observation.state",
        task_key="task",
        num_bins=256,
        max_state_dim=None,
        normalization_mode=normalization_mode,
        statistics=state_stats,
    ))

    num_images = backbone_cfg.get("num_images", 2)
    image_mask = backbone_cfg.get("image_mask", None) or [True] * num_images
    max_token_len = backbone_cfg.get("max_token_len", 200)
    tokenizer_path = (
        getattr(args, "tokenizer_path", None)
        or backbone_cfg.get("tokenizer_name", "")
        or os.environ.get("TOKENIZER_PATH", "")
    )

    transforms.append(Pi05CollateImagesTransform(
        image_size=image_size,
        num_images=num_images,
        image_mask=image_mask,
    ))
    transforms.append(Pi05FallbackPromptTransform())
    transforms.append(Pi05TokenizeTransform(
        tokenizer_path=tokenizer_path,
        max_token_len=max_token_len,
    ))


def _append_fast_transforms(transforms, model_cfg, image_size):
    """Append Fast-specific per-sample transforms if model_type is Fast."""
    model_type = model_cfg.get("model_type", "") if hasattr(model_cfg, "get") else ""
    if model_type != "Fast":
        return

    from loongforge.embodied.data.transforms.fast.fast_transform import FastKeyMappingTransform

    transforms.append(FastKeyMappingTransform(image_size=image_size))
