"""
LeRobot VLA Dataset - Decoupled wrappers around lerobot's dataset classes.

Provides:
    - LeRobotVLADataset(LeRobotDataset): Map-style dataset
    - StreamingLeRobotVLADataset(StreamingLeRobotDataset): Iterable streaming dataset
    - build_lerobot_dataset(): Factory function (mirrors lerobot's make_dataset without config chain)

All classes inherit from official lerobot implementations to fully reuse their
capabilities (delta_timestamps, video decoding, task text resolution, episode handling,
streaming shard management) while providing a simplified VLA-friendly interface that
does NOT require lerobot's full config chain (TrainPipelineConfig, PolicyConfig, etc.)

Output format matches LeRobotDataset native __getitem__:
    {
        "observation.images.<name>": tensor [C, H, W] float32 in [0, 1],
        "observation.state": tensor [state_dim] float32,
        "action": tensor [action_horizon, action_dim] float32,
        "action_is_pad": tensor [action_horizon] bool,
        "task": str,
        "task_index": tensor int64,
        "timestamp": tensor float32,
        "frame_index": tensor int64,
        "episode_index": tensor int64,
        "index": tensor int64,
    }
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

logger = logging.getLogger(__name__)

# Aligned with lerobot/datasets/factory.py
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def _read_dataset_info(dataset_root: Path) -> Dict[str, Any]:
    """Read info.json from dataset meta directory."""
    info_file = dataset_root / "meta" / "info.json"
    if info_file.exists():
        with open(info_file) as f:
            return json.load(f)
    return {}


def _build_delta_timestamps(action_horizon: int, fps: int) -> Dict[str, list]:
    """Build delta_timestamps from action_horizon and fps.

    Mirrors lerobot's resolve_delta_timestamps logic for pi05:
        PI05Config.action_delta_indices = list(range(chunk_size))
        delta_timestamps["action"] = [i / fps for i in action_delta_indices]
    """
    return {
        "action": [i / fps for i in range(action_horizon)],
    }


def _apply_imagenet_stats(dataset) -> None:
    """Override camera stats with ImageNet normalization values.

    Mirrors: lerobot/datasets/factory.py make_dataset() use_imagenet_stats logic.
    """
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "camera_keys"):
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)


class LeRobotVLADataset(LeRobotDataset):
    """Decoupled VLA dataset built on top of LeRobotDataset (Map-style).

    Provides a simplified constructor that builds delta_timestamps from
    action_horizon and dataset fps, without needing lerobot's PolicyConfig
    or TrainPipelineConfig.

    Args:
        repo_id: HuggingFace repo_id or local dataset identifier
        root: Path to local LeRobot v3.0 dataset root directory
        action_horizon: Number of future action steps (maps to chunk_size in PI05Config)
        episodes: Specific episode indices to load (None = all)
        image_transforms: Transform applied to visual modalities
        revision: Git revision (branch/tag/commit hash)
        video_backend: Video decoding backend ("torchcodec", "pyav", etc.)
        tolerance_s: Timestamp tolerance for sync checks
        download_videos: Whether to download videos from Hub
        use_imagenet_stats: Override camera stats with ImageNet mean/std
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        action_horizon: int = 50,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        revision: str | None = None,
        video_backend: str = "torchcodec",
        tolerance_s: float = 1e-4,
        download_videos: bool = False,
        use_imagenet_stats: bool = True,
        transform: Callable | None = None,
    ):
        self._action_horizon = action_horizon
        self._use_imagenet_stats = use_imagenet_stats
        self._transform = transform

        # Resolve root for reading info.json before parent __init__
        dataset_root = Path(root) if root is not None else None
        if dataset_root is not None and dataset_root.exists():
            info = _read_dataset_info(dataset_root)
        else:
            info = {}
        fps = info.get("fps", 10)

        # Build delta_timestamps: mirrors resolve_delta_timestamps for pi05
        delta_timestamps = _build_delta_timestamps(action_horizon, fps)

        # Call parent LeRobotDataset.__init__ with aligned parameters
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        # Apply ImageNet stats override (same as make_dataset)
        if use_imagenet_stats:
            _apply_imagenet_stats(self)

        logger.info(
            f"LeRobotVLADataset: repo_id={repo_id}, len={len(self)}, "
            f"action_horizon={action_horizon}, fps={fps}, "
            f"video_backend={video_backend}"
        )

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self._transform is not None:
            data = self._transform(data)
        return data


class StreamingLeRobotVLADataset(StreamingLeRobotDataset):
    """Decoupled VLA dataset built on top of StreamingLeRobotDataset (Iterable).

    Same decoupled interface as LeRobotVLADataset but for streaming mode.
    Useful for large datasets that don't fit in memory.

    Args:
        repo_id: HuggingFace repo_id or local dataset identifier
        root: Path to local LeRobot v3.0 dataset root directory
        action_horizon: Number of future action steps (maps to chunk_size in PI05Config)
        episodes: Specific episode indices to load (None = all)
        image_transforms: Transform applied to visual modalities
        revision: Git revision (branch/tag/commit hash)
        tolerance_s: Timestamp tolerance for sync checks
        use_imagenet_stats: Override camera stats with ImageNet mean/std
        max_num_shards: Number of shards for streaming parallelism
        buffer_size: Shuffle buffer size for streaming
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle data across exhaustions
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        action_horizon: int = 50,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        revision: str | None = None,
        tolerance_s: float = 1e-4,
        use_imagenet_stats: bool = True,
        max_num_shards: int = 16,
        buffer_size: int = 1000,
        seed: int = 42,
        shuffle: bool = True,
        transform: Callable | None = None,
    ):
        self._action_horizon = action_horizon
        self._use_imagenet_stats = use_imagenet_stats
        self._transform = transform

        # Resolve root for reading info.json before parent __init__
        dataset_root = Path(root) if root is not None else None
        if dataset_root is not None and dataset_root.exists():
            info = _read_dataset_info(dataset_root)
        else:
            info = {}
        fps = info.get("fps", 10)

        # Build delta_timestamps: mirrors resolve_delta_timestamps for pi05
        delta_timestamps = _build_delta_timestamps(action_horizon, fps)

        # Call parent StreamingLeRobotDataset.__init__ with aligned parameters
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            max_num_shards=max_num_shards,
            buffer_size=buffer_size,
            seed=seed,
            shuffle=shuffle,
        )

        # Apply ImageNet stats override (same as make_dataset)
        if use_imagenet_stats:
            _apply_imagenet_stats(self)

        logger.info(
            f"StreamingLeRobotVLADataset: repo_id={repo_id}, "
            f"action_horizon={action_horizon}, fps={fps}, "
            f"max_num_shards={max_num_shards}"
        )

    def __iter__(self):
        for data in super().__iter__():
            if self._transform is not None:
                data = self._transform(data)
            yield data


def _build_lerobot_dataset(
    repo_id: str,
    root: str | Path | None = None,
    action_horizon: int = 50,
    streaming: bool = False,
    episodes: list[int] | None = None,
    image_transforms: Callable | None = None,
    revision: str | None = None,
    video_backend: str = "torchcodec",
    tolerance_s: float = 1e-4,
    download_videos: bool = False,
    use_imagenet_stats: bool = True,
    num_workers: int = 4,
    buffer_size: int = 1000,
    seed: int = 42,
    shuffle: bool = True,
) -> LeRobotDataset | StreamingLeRobotDataset:
    """Factory function to create VLA dataset - mirrors lerobot's make_dataset.

    Decoupled version that does NOT require TrainPipelineConfig or PolicyConfig.
    Computes delta_timestamps internally from action_horizon + dataset fps.

    Args:
        repo_id: HuggingFace repo_id or local dataset identifier
        root: Path to local dataset root. If provided, loads from disk directly.
        action_horizon: Number of future action steps (equivalent to PI05Config.chunk_size)
        streaming: If True, returns StreamingLeRobotVLADataset (IterableDataset)
        episodes: Specific episode indices to load (None = all)
        image_transforms: Transform applied to visual modalities
        revision: Git revision (branch/tag/commit hash)
        video_backend: Video decoder backend (only for non-streaming mode)
        tolerance_s: Timestamp tolerance for sync checks
        download_videos: Whether to download videos from Hub (non-streaming only)
        use_imagenet_stats: Override camera stats with ImageNet mean/std
        num_workers: Used as max_num_shards in streaming mode
        buffer_size: Shuffle buffer size (streaming only)
        seed: Random seed (streaming only)
        shuffle: Whether to shuffle (streaming only)

    Returns:
        LeRobotVLADataset or StreamingLeRobotVLADataset instance
    """
    if streaming:
        dataset = StreamingLeRobotVLADataset(
            repo_id=repo_id,
            root=root,
            action_horizon=action_horizon,
            episodes=episodes,
            image_transforms=image_transforms,
            revision=revision,
            tolerance_s=tolerance_s,
            use_imagenet_stats=use_imagenet_stats,
            max_num_shards=num_workers,
            buffer_size=buffer_size,
            seed=seed,
            shuffle=shuffle,
        )
    else:
        dataset = LeRobotVLADataset(
            repo_id=repo_id,
            root=root,
            action_horizon=action_horizon,
            episodes=episodes,
            image_transforms=image_transforms,
            revision=revision,
            video_backend=video_backend,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            use_imagenet_stats=use_imagenet_stats,
        )

    return dataset


def build_lerobot_dataset(model_cfg, args):
    """Build lerobot-based VLA dataset from model config and CLI args."""
    dataset_path = getattr(args, "dataset_path", None)
    if not dataset_path:
        raise ValueError("Must specify --dataset-path")

    dataset_path = Path(dataset_path)
    repo_id = dataset_path.name

    action_cfg = model_cfg.get("action_model", {}) if hasattr(model_cfg, "get") else {}
    action_horizon = getattr(args, "action_horizon", action_cfg.get("action_horizon", 50))

    return _build_lerobot_dataset(
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
