# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
LoongForge VLA Data Engine

Data loading for VLA (Vision-Language-Action) training.

Public API:
    - build_dataloader(model_cfg, args, ctx): Factory that builds DataLoader with
      model-specific preprocessor as collate_fn
    - save_dataset_statistics: Save stats to JSON
    - BasePreprocessor / PreparedBatch: Base classes for extension
    - register_preprocessor / get_preprocessor: Registry API

DataLoader output:
    PreparedBatch subclass (CPU tensors, model-specific fields).

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
from torch.utils.data import Dataset, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from loongforge.embodied.distributed import DistributedContext
from loongforge.embodied.data.transforms import (
    BasePreprocessor,
    PreparedBatch,
)
from loongforge.embodied.data.transforms.pipeline import build_transforms_from_args
from loongforge.embodied.data.transforms.collator import build_preprocessor

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

    # Build dataset (without transform first to get stats)
    dataset = _build_dataset(model_cfg, args, module)

    # Get dataset stats
    dataset_stats = {}
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "stats"):
        dataset_stats = dataset.meta.stats

    # Build per-sample transforms and inject into dataset
    transform = build_transforms_from_args(model_cfg, args, dataset, dataset_stats)
    if transform is not None:
        if hasattr(dataset, "_transform"):
            dataset._transform = transform
        elif isinstance(dataset, IterableDataset):
            dataset = _TransformedIterableDataset(dataset, transform)
        else:
            dataset = _TransformedMapDataset(dataset, transform)

    # Build preprocessor (collate_fn)
    preprocessor = _build_preprocessor(model_cfg, args)

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


def _build_preprocessor(model_cfg, args):
    """Build batch-level preprocessor (collate_fn) from the registry.

    Resolves model_type to a registered preprocessor name, then
    instantiates via from_config.
    """
    model_type = model_cfg.get("model_type", "") if hasattr(model_cfg, "get") else ""
    if not model_type:
        model_type = "dummy"

    preprocessor = build_preprocessor(model_type, model_cfg)

    # Override fast_tokenizer_path with CLI --tokenizer-path if provided
    if hasattr(preprocessor, "fast_tokenizer_path") and getattr(args, "tokenizer_path", None):
        preprocessor.fast_tokenizer_path = args.tokenizer_path

    logger.info(f"Using preprocessor: {model_type}")
    return preprocessor


def _build_dataset(model_cfg, args, module: str):
    """Build dataset instance based on dataloader_module."""
    if module == "lerobot_datasets":
        from .datasets.lerobot_dataset import build_lerobot_dataset
        return build_lerobot_dataset(model_cfg, args)

    elif module == "rlds_datasets":
        from .datasets.rlds_dataset import build_rlds_dataset
        return build_rlds_dataset(model_cfg, args)

    elif module == "hdf5_datasets":
        from .datasets.hdf5_dataset import build_hdf5_dataset
        return build_hdf5_dataset(model_cfg, args)

    elif module == "dummy_datasets":
        from .datasets.dummy_dataset import build_dummy_dataset
        return build_dummy_dataset(model_cfg, args)

    else:
        raise ValueError(
            f"Unknown dataloader_module: '{module}'. "
            f"Supported: lerobot_datasets, rlds_datasets, hdf5_datasets, dummy_datasets"
        )


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


class _TransformedMapDataset(Dataset):
    """Wrapper that applies a transform to a map-style dataset."""

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data = self._dataset[idx]
        return self._transform(data)

    def __getattr__(self, name):
        return getattr(self._dataset, name)


class _TransformedIterableDataset(IterableDataset):
    """Wrapper that applies a transform to an iterable dataset."""

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __iter__(self):
        for data in self._dataset:
            yield self._transform(data)

    def __getattr__(self, name):
        return getattr(self._dataset, name)


__all__ = [
    "build_dataloader",
    "save_dataset_statistics",
    "BasePreprocessor",
    "PreparedBatch",
]
