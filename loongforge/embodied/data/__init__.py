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
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler
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


class _BlockShardSampler(Sampler[int]):
    """Distribute contiguous (non-strided) batches across data-parallel ranks.

    For each global "batch slot" of ``batch_size`` consecutive indices in the
    shuffled order, this sampler assigns the **entire** batch to a single
    data-parallel rank in round-robin fashion. As a result, every rank's
    micro-batch is a contiguous slice of the shuffled index sequence, not a
    strided one.

    This is the same semantics implemented by HuggingFace Accelerate's
    ``BatchSamplerShard(split_batches=False)`` and matches.

    Comparison with ``torch.utils.data.distributed.DistributedSampler``
    -----------------------------------------------------------------
    Both samplers consume the same global index sequence but split it across
    ranks differently. To make the difference easy to read, the example below
    uses ``shuffle=False`` so the global index sequence is just
    ``[0, 1, 2, ..., N-1]``.

    Example parallelism setup (single node, DDP-only, no TP / PP)::

        world_size                     = 8      # total processes launched by torchrun
        dp_size                        = 8      # data-parallel replicas (= world_size here)
        mbs (per-device batch size)    = 2      # --per-device-batch-size
        grad_accum   (--gradient-accumulation-steps) = 1
        gbs (global batch size)        = 16


    Global index sequence with ``shuffle=False``::

        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    Grouping into blocks of size ``batch_size = mbs = 2`` (each block is one
    rank's micro-batch)::

        block 0: [ 0,  1]     block 4: [ 8,  9]
        block 1: [ 2,  3]     block 5: [10, 11]
        block 2: [ 4,  5]     block 6: [12, 13]
        block 3: [ 6,  7]     block 7: [14, 15]

    ``_BlockShardSampler`` (this class) — round-robin **blocks** (whole
    micro-batches) across DP ranks. block ``b`` goes to rank ``b % dp_size``::

        rank 0 iterates:  0,  1     # block 0 (opt step 0)
        rank 1 iterates:  2,  3     # block 1 (opt step 0)
        rank 2 iterates:  4,  5     # block 2 (opt step 0)
        rank 3 iterates:  6,  7     # block 3 (opt step 0)
        rank 4 iterates:  8,  9     # block 4 (opt step 0)
        rank 5 iterates: 10, 11     # block 5 (opt step 0)
        rank 6 iterates: 12, 13     # block 6 (opt step 0)
        rank 7 iterates: 14, 15     # block 7 (opt step 0)

    ``DistributedSampler`` — round-robin **samples** (stride sharding) across
    DP ranks; PyTorch's built-in ``BatchSampler`` afterwards groups each rank's
    per-sample stream into micro-batches of size ``mbs``::

        rank 0 iterates:  0,  8     # micro-batch 0 = [ 0,  8]
        rank 1 iterates:  1,  9     # micro-batch 0 = [ 1,  9]
        rank 2 iterates:  2, 10     # micro-batch 0 = [ 2, 10]
        rank 3 iterates:  3, 11     # micro-batch 0 = [ 3, 11]
        rank 4 iterates:  4, 12     # micro-batch 0 = [ 4, 12]
        rank 5 iterates:  5, 13     # micro-batch 0 = [ 5, 13]
        rank 6 iterates:  6, 14     # micro-batch 0 = [ 6, 14]
        rank 7 iterates:  7, 15     # micro-batch 0 = [ 7, 15]
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        length = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(length, generator=generator).tolist()
        else:
            indices = list(range(length))

        if self.drop_last:
            usable = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:usable]

        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
            if len(indices[start : start + self.batch_size]) == self.batch_size or not self.drop_last
        ]
        for batch_idx, batch in enumerate(batches):
            if batch_idx % self.num_replicas == self.rank:
                yield from batch

    def __len__(self) -> int:
        full_batches, remainder = divmod(len(self.dataset), self.batch_size)
        total_batches = full_batches if self.drop_last or remainder == 0 else full_batches + 1
        local_batches = (total_batches + self.num_replicas - 1 - self.rank) // self.num_replicas
        if not local_batches:
            return 0
        if self.drop_last or remainder == 0 or (local_batches - 1) * self.num_replicas + self.rank < full_batches:
            return local_batches * self.batch_size
        return (local_batches - 1) * self.batch_size + remainder

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index for sampling."""
        self.epoch = epoch

    def state_dict(self) -> dict:
        """Return state_dict for checkpointing."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state_dict from checkpoint."""
        self.epoch = int(state_dict.get("epoch", 0))


class _SeedWorkerInit:
    """Picklable worker initializer for spawn/forkserver DataLoader workers."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


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
    mp_context = getattr(args, "dataloader_multiprocessing_context", None) or ("spawn" if num_workers > 0 else None)

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
            prefetch_factor=2 if num_workers > 0 else None,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0 and mp_context == "spawn",
        )
    else:
        sampler = None
        shuffle = True
        sampler_mode = getattr(args, "distributed_sampler_mode", "cyclic")
        generator = torch.Generator().manual_seed(seed)
        use_distributed_sampler = ctx.is_distributed and ctx.world_size > 1
        if use_distributed_sampler:
            if sampler_mode == "block":
                sampler = _BlockShardSampler(
                    dataset,
                    batch_size=batch_size,
                    num_replicas=ctx.world_size,
                    rank=ctx.rank,
                    shuffle=shuffle,
                    seed=seed,
                    drop_last=False,
                )
            else:
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
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=_SeedWorkerInit(seed) if num_workers > 0 else None,
            generator=generator,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0 and mp_context == "spawn",
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

    preprocessor = build_preprocessor(model_type, model_cfg, args=args)

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
