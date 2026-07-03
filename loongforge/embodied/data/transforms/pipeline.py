# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLA transforms pipeline builder.

Pipeline builder:
    - convert_stats: Convert dataset statistics to numpy format
    - build_transforms_from_args: Build per-sample transforms from config + CLI training_args
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
    data_cfg,
    training_args,
    dataset,
    dataset_stats,
) -> Optional[ComposedTransform]:
    """Build per-sample transforms from typed ModelConfig + DataConfig (+ TrainingArgs).

    Args:
        model_cfg: typed ModelConfig (model structure + shared fields).
        data_cfg: typed DataConfig (data-processing fields).
        training_args: TrainingArgs (generic CLI params).
        dataset: The dataset instance (used to discover image keys).
        dataset_stats: Dict of normalization statistics from dataset.meta.stats.

    Returns:
        ComposedTransform or None if not applicable.
    """
    model_type = model_cfg.model_type

    # Shared fields → ModelConfig; data-processing fields → DataConfig.
    image_size = data_cfg.image_size
    action_horizon = model_cfg.action_horizon
    max_action_dim = model_cfg.max_action_dim
    normalization_mode = data_cfg.normalization_mode

    # Discover image keys from first sample
    try:
        first_sample = dataset[0]
        image_keys = sorted(k for k in first_sample.keys() if k.startswith("observation.images."))
    except Exception:
        image_keys = []

    transforms = []

    # 1. Image transform (configurable via DataConfig)
    if image_keys and data_cfg.use_image_transform:
        transforms.append(ImageTransform(
            apply_to=image_keys,
            image_size=image_size,
            resize_strategy=data_cfg.image_resize_strategy,
            normalize_mode=data_cfg.image_normalize_mode,
        ))

    # 2. Action transform (configurable via DataConfig)
    if data_cfg.use_action_transform:
        action_stats = (
            convert_stats(dataset_stats.get("action"))
            if dataset_stats and data_cfg.action_use_statistics
            else None
        )
        transform_action_horizon = (
            data_cfg.action_transform_horizon
            if data_cfg.action_transform_horizon is not None
            else action_horizon
        )
        transform_max_action_dim = (
            data_cfg.action_transform_max_action_dim
            if data_cfg.action_transform_max_action_dim is not None
            else max_action_dim
        )
        transforms.append(ActionTransform(
            apply_to=data_cfg.action_apply_to,
            action_horizon=transform_action_horizon,
            max_action_dim=transform_max_action_dim,
            normalization_mode=normalization_mode,
            statistics=action_stats,
            padding_strategy=data_cfg.action_padding_strategy,
        ))

    # 3. Pi05-specific transforms: state discretization, collate images, fallback prompt, tokenize
    if model_type == "pi05":
        _append_pi05_transforms(transforms, model_cfg, data_cfg, training_args, dataset_stats, image_size)

    # 4. GR00T-N1.6-specific transforms: prompt fallback and feature assembly
    elif model_type == "Gr00tN1d6":
        _append_groot_n1_6_transforms(transforms, model_cfg, data_cfg, dataset_stats, dataset)

    if not transforms:
        return None

    return ComposedTransform(transforms)


def _append_pi05_transforms(transforms, model_cfg, data_cfg, training_args, dataset_stats, image_size):
    """Append Pi05-specific per-sample transforms."""
    import os
    from loongforge.embodied.data.transforms.pi05.pi05_transform import (
        StateDiscretizationTransform,
        Pi05CollateImagesTransform,
        Pi05FallbackPromptTransform,
        Pi05TokenizeTransform,
    )

    normalization_mode = data_cfg.normalization_mode
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

    num_images = data_cfg.num_images
    image_mask = data_cfg.image_mask or [True] * num_images
    max_token_len = data_cfg.max_token_len
    tokenizer_path = (
        training_args.tokenizer_path
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


def _append_groot_n1_6_transforms(transforms, model_cfg, data_cfg, dataset_stats, dataset):
    """Append GR00T-N1.6-specific per-sample transforms."""
    from loongforge.embodied.data.transforms.groot_n1_6.groot_transform import (
        GrootN1d6FeatureTransform,
        GrootPromptTransform,
    )

    preprocess_mode = data_cfg.groot_preprocess_mode
    if preprocess_mode != "sample":
        raise ValueError(
            "groot_preprocess_mode must be 'sample' after GR00T preprocessing "
            "was moved into per-sample transforms; "
            f"got {preprocess_mode!r}"
        )

    transforms.append(GrootPromptTransform())
    transforms.append(GrootN1d6FeatureTransform(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        dataset_stats=dataset_stats,
        dataset=dataset,
    ))
