# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
LoongForge VLA Data Engine

Data loading for VLA (Vision-Language-Action) training.

Public API:
    - build_dataloader(model_cfg, args, ctx): Factory that builds DataLoader with
      model-specific preprocessor as collate_fn
    - build_lerobot_dataset: Build raw dataset (without preprocessor)
    - save_dataset_statistics: Save stats to JSON
    - BasePreprocessor / PreparedBatch: Base classes for extension
    - register_preprocessor / get_preprocessor: Registry API

DataLoader output:
    Pi05PreparedBatch (CPU tensors):
        .images_list: List[Tensor (B, 3, H, W)]
        .img_masks:   List[Tensor (B,) bool]
        .input_ids:   Tensor (B, seq_len)
        .attention_mask: Tensor (B, seq_len) bool
        .actions:     Tensor (B, T, D)

    Usage:
        for batch in dataloader:
            batch = batch.to(device)
            output = model.forward(batch)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from loongforge.embodied.distributed import DistributedContext
from loongforge.embodied.data.transforms import (
    BasePreprocessor,
    PreparedBatch,
    Pi05Preprocessor,
    Pi05PreparedBatch,
    convert_stats,
    StateDiscretizationTransform,
)
from loongforge.embodied.data.transforms.pipeline import build_transforms_from_args

logger = logging.getLogger(__name__)


def build_dataloader(model_cfg, args, ctx: DistributedContext) -> StatefulDataLoader:
    """Build DataLoader with model-specific preprocessor as collate_fn.

    The returned DataLoader yields PreparedBatch objects (CPU tensors).
    Call batch.to(device) before passing to model.forward().

    Args:
        model_cfg: OmegaConf/dict model config (backbone, action_model at top level)
        args: CLI args namespace (dataset path, batch size, etc.)
        ctx: DistributedContext

    Returns:
        torch.utils.data.DataLoader (yielding PreparedBatch subclass)
    """
    module = getattr(args, "dataloader_module", "lerobot_datasets")
    batch_size = args.per_device_batch_size
    num_workers = getattr(args, "num_workers", 4)

    # Build dataset
    dataset = _build_dataset(model_cfg, args, module)

    # Get dataset stats
    dataset_stats = {}
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "stats"):
        dataset_stats = dataset.meta.stats

    # Build preprocessor (collate_fn)
    preprocessor = _build_preprocessor(model_cfg, args, dataset, dataset_stats)

    # Save statistics
    if ctx.is_main:
        output_dir = getattr(args, "output_dir", "")
        if output_dir and dataset_stats:
            stats_path = os.path.join(output_dir, "dataset_statistics.json")
            save_dataset_statistics(dataset_stats, stats_path)

    # Build DataLoader
    seed = getattr(args, "seed", 0) or 0
    if isinstance(dataset, IterableDataset):
        dl = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=preprocessor,
            pin_memory=True,
            drop_last=False,
        )
    else:
        sampler = None
        shuffle = True
        if ctx.is_distributed:
            sampler = StatefulDistributedSampler(
                dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=False,
            )
            shuffle = False

        dl = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=preprocessor,
            pin_memory=True,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
        dl.steps_per_epoch = len(dl)

    return dl


def _build_preprocessor(model_cfg, args, dataset, dataset_stats):
    """Build Pi05Preprocessor with unified transforms pipeline.

    Constructs per-sample transforms (image + action + state discretization) via
    build_transforms_from_args and injects them into the preprocessor. The preprocessor
    only handles batch-level collation (stack + tokenize).
    """
    backbone_cfg = model_cfg.get("backbone", {}) if hasattr(model_cfg, "get") else {}

    tokenizer_path = (
        getattr(args, "tokenizer_path", None)
        or backbone_cfg.get("tokenizer_name", "")
        or os.environ.get("TOKENIZER_PATH", "")
    )

    norm_mode = getattr(args, "normalization_mode", "q99")
    max_token_len = backbone_cfg.get("max_token_len", 200)

    # Build state transform as pi05-specific extra
    state_stats = convert_stats(dataset_stats.get("observation.state")) if dataset_stats else None
    state_transform = StateDiscretizationTransform(
        apply_to=["prompt"],
        state_key="observation.state",
        task_key="task",
        num_bins=256,
        max_state_dim=None,
        normalization_mode=norm_mode,
        statistics=state_stats,
    )

    # Build unified per-sample transforms pipeline (image + action + state)
    transform = build_transforms_from_args(
        model_cfg, args, dataset, dataset_stats,
        extra_transforms=[state_transform],
    )

    preprocessor = Pi05Preprocessor(
        image_size=getattr(args, "image_size", backbone_cfg.get("image_size", 224)),
        num_images=backbone_cfg.get("num_images", 2),
        image_mask=backbone_cfg.get("image_mask", None),
        max_token_len=max_token_len,
        tokenizer_path=tokenizer_path,
        transform=transform,
    )

    logger.info(f"Using preprocessor: Pi05Preprocessor (transforms pipeline, max_token_len={max_token_len})")
    return preprocessor


def _build_dataset(model_cfg, args, module: str):
    """Build dataset instance based on dataloader_module."""
    if module == "lerobot_datasets":
        return _build_lerobot_dataset(model_cfg, args)
    else:
        raise ValueError(
            f"Unknown dataloader_module: '{module}'. "
            f"Supported: lerobot_datasets"
        )


def _build_lerobot_dataset(model_cfg, args):
    """Build lerobot-based VLA dataset."""
    from loongforge.embodied.data.datasets.lerobot_dataset import build_lerobot_dataset

    dataset_path = getattr(args, "dataset_path", None)
    if not dataset_path:
        raise ValueError("Must specify --dataset-path")

    dataset_path = Path(dataset_path)
    repo_id = dataset_path.name

    action_cfg = model_cfg.get("action_model", {}) if hasattr(model_cfg, "get") else {}
    action_horizon = getattr(args, "action_horizon", action_cfg.get("action_horizon", 50))

    dataset = build_lerobot_dataset(
        repo_id=repo_id,
        root=str(dataset_path),
        action_horizon=action_horizon,
        streaming=False,
        episodes=None,
        video_backend="torchcodec",
        tolerance_s=1e-4,
        download_videos=False,
        use_imagenet_stats=True,
        num_workers=getattr(args, "num_workers", 4),
    )

    return dataset


def save_dataset_statistics(dataset_statistics: Dict, output_path):
    """Save dataset_statistics.json for action denormalization during inference."""
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    serializable = {}
    for key, stats in dataset_statistics.items():
        serializable[key] = {}
        for stat_name, stat_val in stats.items():
            if isinstance(stat_val, torch.Tensor):
                serializable[key][stat_name] = stat_val.cpu().tolist()
            elif isinstance(stat_val, np.ndarray):
                serializable[key][stat_name] = stat_val.tolist()
            else:
                serializable[key][stat_name] = stat_val

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved dataset statistics to {output_path}")


def _build_dataset(model_cfg, args, module: str):
    """Build dataset instance based on dataloader_module."""
    if module == "lerobot_datasets":
        return _build_lerobot_dataset(model_cfg, args)

    elif module == "rlds_datasets":
        from .datasets.rlds_dataset import build_rlds_dataset

        return build_rlds_dataset(model_cfg, args)

    elif module == "hdf5_datasets":
        from .datasets.hdf5_dataset import build_hdf5_dataset

        return build_hdf5_dataset(model_cfg, args)

    elif module == "dummy_datasets":
        from .datasets.dummy_dataset import build_dummy_dataset

        return build_dummy_dataset(model_cfg, args)

    elif module == "mock_lerobotv2_datasets":
        from .datasets.mock_lerobot_v2_dataset import build_mock_lerobot_v2_dataset

        return build_mock_lerobot_v2_dataset(model_cfg, args)

    else:
        raise ValueError(
            f"Unknown dataloader_module: '{module}'. "
            f"Supported: lerobot_datasets, rlds_datasets, hdf5_datasets, dummy_datasets, mock_lerobotv2_datasets"
        )


__all__ = [
    "build_dataloader",
    "build_lerobot_dataset",
    "save_dataset_statistics",
    "BasePreprocessor",
    "PreparedBatch",
    "Pi05Preprocessor",
    "Pi05PreparedBatch",
]
