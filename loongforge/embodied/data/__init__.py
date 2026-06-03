# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Data loading framework with built-in DistributedSampler.

Provides:
  - build_dataloader(): Unified factory, dispatches to dataset backends
  - Supports: lerobot_datasets, rlds_datasets, hdf5_datasets, dummy_datasets

Sample output format (__getitem__):
    {
        "image": [PIL.Image, ...],
        "lang": str,
        "action": np.ndarray [action_horizon, action_dim],
        "state": np.ndarray | None,
    }
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from loongforge.embodied.distributed import DistributedContext

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Default collate: return list directly, batching handled by model."""
    return batch


def build_dataloader(model_cfg, args, ctx: DistributedContext) -> DataLoader:
    """
    Unified dataloader factory with native DistributedSampler.

    Args:
        model_cfg: OmegaConf model config (for transforms that need model info)
        args: CLI args namespace (contains dataset path, batch size, etc.)
        ctx: DistributedContext
    """
    module = args.dataloader_module
    batch_size = args.per_device_batch_size
    num_workers = args.num_workers

    dataset = _build_dataset(model_cfg, args, module)

    # IterableDataset (e.g. RLDS streaming) does not support sampler/shuffle
    if isinstance(dataset, IterableDataset):
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
    else:
        # DistributedSampler for multi-GPU (map-style datasets only)
        sampler = None
        shuffle = True
        if ctx.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
            shuffle = False

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    # Save dataset statistics if available
    if ctx.is_main and hasattr(dataset, "dataset_statistics"):
        output_dir = getattr(args, "output_dir", "")
        if output_dir:
            stats_path = os.path.join(output_dir, "dataset_statistics.json")
            save_dataset_statistics(dataset.dataset_statistics, stats_path)

    # Append Pi05StateTransform if discrete_state_input is enabled
    _maybe_append_pi05_state_transform(model_cfg, args, dl)

    return dl


def save_dataset_statistics(dataset_statistics: Dict, output_path):
    """Save dataset_statistics.json for action denormalization during inference."""
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    serializable = {}
    for key, stats in dataset_statistics.items():
        serializable[key] = {}
        for stat_name, stat_val in stats.items():
            if isinstance(stat_val, np.ndarray):
                serializable[key][stat_name] = stat_val.tolist()
            else:
                serializable[key][stat_name] = stat_val

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved dataset statistics to {output_path}")


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


def _maybe_append_pi05_state_transform(model_cfg, args, dataloader: DataLoader):
    """Conditionally append Pi05StateTransform to dataset's transform pipeline."""
    if not getattr(args, "discrete_state_input", False):
        return

    from .transforms import ComposedTransform, Pi05StateTransform

    action_model_cfg = model_cfg.get("framework", {}).get("action_model", {})
    max_state_dim = action_model_cfg.get("max_state_dim", 32)

    dataset = dataloader.dataset
    transform = Pi05StateTransform(apply_to=["lang"], max_state_dim=max_state_dim)

    if hasattr(dataset, "transform") and dataset.transform is not None:
        if isinstance(dataset.transform, ComposedTransform):
            dataset.transform.transforms.append(transform)
        else:
            dataset.transform = ComposedTransform([dataset.transform, transform])
    else:
        dataset.transform = transform
