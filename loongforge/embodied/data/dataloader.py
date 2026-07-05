# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DataLoader factory for embodied VLA training."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from loongforge.embodied.data.transforms.collator import build_preprocessor
from loongforge.embodied.data.transforms.pipeline import build_transforms_from_args
from loongforge.embodied.distributed import DistributedContext

logger = logging.getLogger(__name__)


class _SeedWorkerInit:
    """Picklable worker initializer for spawn/forkserver DataLoader workers."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


class _BlockShardSampler(Sampler[int]):
    """Distribute contiguous batches across data-parallel ranks."""

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
        """Set the epoch index for deterministic shuffling."""
        self.epoch = epoch

    def state_dict(self) -> dict:
        """Return checkpointable sampler state."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load checkpointed sampler state."""
        self.epoch = int(state_dict.get("epoch", 0))


class _TransformedMapDataset(Dataset):
    """Wrap a map-style dataset and apply a per-sample transform."""

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
    """Wrap an iterable dataset and apply a per-sample transform."""

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __iter__(self):
        for data in self._dataset:
            yield self._transform(data)

    def __getattr__(self, name):
        return getattr(self._dataset, name)


def build_dataloader(model_cfg, data_cfg, training_args, ctx: DistributedContext) -> StatefulDataLoader:
    """Build DataLoader with model-specific preprocessor as collate_fn.

    The returned DataLoader yields PreparedBatch objects on CPU. Call
    ``batch.to(device)`` before passing a batch to ``model.forward``.
    """
    dataset_format = training_args.dataset_format
    batch_size = training_args.per_device_batch_size
    num_workers = training_args.num_workers
    mp_context = training_args.dataloader_multiprocessing_context or ("spawn" if num_workers > 0 else None)

    dataset = _build_dataset(model_cfg, data_cfg, training_args, dataset_format)
    dataset_stats = _get_dataset_stats(dataset)

    transform = build_transforms_from_args(model_cfg, data_cfg, training_args, dataset, dataset_stats)
    if transform is not None:
        dataset = _apply_transform(dataset, transform)

    preprocessor = _build_preprocessor(model_cfg, data_cfg, training_args, dataset_stats, dataset)

    return _build_stateful_dataloader(
        dataset=dataset,
        preprocessor=preprocessor,
        batch_size=batch_size,
        num_workers=num_workers,
        mp_context=mp_context,
        training_args=training_args,
        ctx=ctx,
    )


def _get_dataset_stats(dataset):
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "stats"):
        return dataset.meta.stats
    return {}


def _apply_transform(dataset, transform):
    if hasattr(dataset, "_transform"):
        dataset._transform = transform
        return dataset
    if isinstance(dataset, IterableDataset):
        return _TransformedIterableDataset(dataset, transform)
    return _TransformedMapDataset(dataset, transform)


def _build_stateful_dataloader(
    *,
    dataset,
    preprocessor,
    batch_size: int,
    num_workers: int,
    mp_context,
    training_args,
    ctx: DistributedContext,
) -> StatefulDataLoader:
    seed = training_args.seed
    seed_workers = training_args.dataloader_seed_workers
    if isinstance(dataset, IterableDataset):
        return StatefulDataLoader(
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

    sampler = None
    shuffle = True
    sampler_mode = training_args.distributed_sampler_mode
    generator = torch.Generator().manual_seed(seed) if seed_workers else None
    if ctx.is_distributed and ctx.world_size > 1:
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
        worker_init_fn=_SeedWorkerInit(seed) if seed_workers and num_workers > 0 else None,
        generator=generator,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0 and mp_context == "spawn",
    )
    dl.steps_per_epoch = len(dl)
    return dl


def _build_preprocessor(model_cfg, data_cfg, training_args, dataset_stats, dataset):
    """Build batch-level preprocessor from the collator registry."""
    model_type = model_cfg.model_type or "dummy"
    preprocessor = build_preprocessor(
        model_type,
        model_cfg,
        data_cfg,
        training_args=training_args,
        dataset_stats=dataset_stats,
        dataset=dataset,
    )

    logger.info(f"Using preprocessor: {model_type}")
    return preprocessor


def _build_dataset(model_cfg, data_cfg, training_args, dataset_format: str):
    """Build dataset instance based on ``--dataset-format``."""
    if dataset_format == "lerobot_datasets":
        from .datasets.lerobot_dataset import build_lerobot_dataset

        return build_lerobot_dataset(model_cfg, data_cfg, training_args)

    if dataset_format == "rlds_datasets":
        from .datasets.rlds_dataset import build_rlds_dataset

        return build_rlds_dataset(model_cfg, data_cfg, training_args)

    if dataset_format == "hdf5_datasets":
        from .datasets.hdf5_dataset import build_hdf5_dataset

        return build_hdf5_dataset(model_cfg, data_cfg, training_args)

    if dataset_format == "dummy_datasets":
        from .datasets.dummy_dataset import build_dummy_dataset

        return build_dummy_dataset(model_cfg, data_cfg, training_args)

    raise ValueError(
        f"Unknown dataset_format: '{dataset_format}'. "
        f"Supported: lerobot_datasets, rlds_datasets, hdf5_datasets, dummy_datasets"
    )
