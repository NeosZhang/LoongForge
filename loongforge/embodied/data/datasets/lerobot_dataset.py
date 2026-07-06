"""
LeRobot VLA Dataset - Multi-version support for LeRobot dataset formats.

Provides:
    - LeRobotV3Dataset(LeRobotDataset): v3.0 Map-style dataset (wraps official lerobot)
    - StreamingLeRobotV3Dataset(StreamingLeRobotDataset): v3.0 Iterable streaming dataset
    - LeRobotV2Dataset(Dataset): v2.0/v2.1 format (no lerobot lib dependency)
    - build_lerobot_dataset(): Factory function with version dispatch

Output format (all versions):
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

logger = logging.getLogger(__name__)


def _read_dataset_info(dataset_root: Path) -> Dict[str, Any]:
    """Read info.json from dataset meta directory."""
    info_file = dataset_root / "meta" / "info.json"
    if info_file.exists():
        with open(info_file) as f:
            return json.load(f)
    return {}


def _build_delta_timestamps(action_horizon: int, fps: int) -> Dict[str, list]:
    """Build delta_timestamps from action_horizon and fps."""
    return {
        "action": [i / fps for i in range(action_horizon)],
    }


class LeRobotV3Dataset(LeRobotDataset):
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
        transform: Callable | None = None,
    ):
        self._action_horizon = action_horizon
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

        logger.info(
            f"LeRobotV3Dataset: repo_id={repo_id}, len={len(self)}, "
            f"action_horizon={action_horizon}, fps={fps}, "
            f"video_backend={video_backend}"
        )

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self._transform is not None:
            data = self._transform(data)
        return data


class LeRobotV2Dataset(Dataset):
    """LeRobot v2.0/v2.1 format dataset (does not depend on the official lerobot library).

    v2.0/v2.1 layout:
      - meta/info.json: fps, chunks_size, video keys, etc.
      - meta/episodes.jsonl: one episode per line  {episode_index, length, ...}
      - meta/tasks.jsonl: one task per line  {task_index, task}
      - data/{chunk}/episode_{idx:06d}.parquet: one parquet file per episode
      - videos/{key}/{chunk}/episode_{idx:06d}.mp4: video files

    Output format is aligned with the v3.0 LeRobotV3Dataset.
    """

    def __init__(
        self,
        root: str | Path,
        action_horizon: int = 50,
        episodes: list[int] | None = None,
        video_backend: str = "torchcodec",
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self._transform = transform

        # Load metadata
        self.info = _read_dataset_info(self.root)
        self.fps = self.info.get("fps", 10)
        self.chunks_size = self.info.get("chunks_size", 1000)

        # Identify keys from features (video dtype entries)
        features = self.info.get("features", {})
        self._video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        self._state_key = "observation.state"
        self._video_path_tpl = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )

        # Load episodes and tasks
        self._episodes = self._load_episodes()
        self._tasks = self._load_tasks()

        # Filter episodes if specified
        if episodes is not None:
            self._episodes = [e for e in self._episodes if e["episode_index"] in episodes]

        # Build flat step index: list of (episode_index, local_step_idx)
        self._step_index: List[Tuple[int, int]] = []
        for ep in self._episodes:
            ep_idx = ep["episode_index"]
            for step in range(ep["length"]):
                self._step_index.append((ep_idx, step))

        # Cache for loaded episode data
        self._cached_ep_idx: int | None = None
        self._cached_ep_data: pd.DataFrame | None = None

        # Build stats (lazy)
        self._stats: Dict | None = None

        logger.info(
            f"LeRobotV2Dataset: root={root}, episodes={len(self._episodes)}, "
            f"total_steps={len(self._step_index)}, action_horizon={action_horizon}, fps={self.fps}"
        )

    @property
    def meta(self):
        """Compatibility shim for stats access (matches v3.0 interface)."""
        class _Meta:
            pass
        m = _Meta()
        m.stats = self.stats
        m.camera_keys = self._video_keys
        return m

    @property
    def stats(self) -> Dict:
        """Load or compute dataset statistics."""
        if self._stats is None:
            stats_path = self.root / "meta" / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    raw = json.load(f)
                self._stats = {
                    k: {sk: torch.tensor(sv) for sk, sv in v.items()}
                    for k, v in raw.items()
                }
            else:
                self._stats = {}
        return self._stats

    def _load_episodes(self) -> List[Dict]:
        """Parse meta/episodes.jsonl."""
        ep_file = self.root / "meta" / "episodes.jsonl"
        episodes = []
        with open(ep_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes

    def _load_tasks(self) -> Dict[int, str]:
        """Parse meta/tasks.jsonl → {task_index: task_text}."""
        task_file = self.root / "meta" / "tasks.jsonl"
        tasks = {}
        if task_file.exists():
            with open(task_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        tasks[obj["task_index"]] = obj["task"]
        return tasks

    def _get_episode_data(self, episode_index: int) -> pd.DataFrame:
        """Load parquet for a single episode (with caching)."""
        if self._cached_ep_idx == episode_index:
            return self._cached_ep_data

        chunk = episode_index // self.chunks_size
        parquet_path = self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        if not parquet_path.exists():
            parquet_path = self.root / "data" / f"{chunk:06d}" / f"episode_{episode_index:06d}.parquet"

        self._cached_ep_data = pd.read_parquet(parquet_path)
        self._cached_ep_idx = episode_index
        return self._cached_ep_data

    def _decode_video_frame(self, video_key: str, episode_index: int, frame_index: int) -> torch.Tensor:
        """Decode a single video frame. Returns tensor [C, H, W] float32 in [0, 1]."""
        episode_chunk = episode_index // self.chunks_size
        video_path = self.root / self._video_path_tpl.format(
            episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index,
        )

        timestamp = frame_index / self.fps

        from loongforge.embodied.data.video_backends import decode_video_frame
        frame = decode_video_frame(str(video_path), timestamp, backend=self.video_backend)
        return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    def __len__(self) -> int:
        return len(self._step_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        episode_index, step_idx = self._step_index[index]
        ep_data = self._get_episode_data(episode_index)

        sample = {}

        # Video frames
        for vkey in self._video_keys:
            out_key = vkey if vkey.startswith("observation.images.") else f"observation.images.{vkey}"
            if vkey in ep_data.columns:
                img_data = ep_data.iloc[step_idx][vkey]
                if isinstance(img_data, (list, np.ndarray)):
                    arr = np.array(img_data, dtype=np.uint8)
                    if arr.ndim == 3:  # (H, W, C)
                        sample[out_key] = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                    else:
                        sample[out_key] = self._decode_video_frame(vkey, episode_index, step_idx)
                else:
                    sample[out_key] = self._decode_video_frame(vkey, episode_index, step_idx)
            else:
                sample[out_key] = self._decode_video_frame(vkey, episode_index, step_idx)

        # State
        if self._state_key in ep_data.columns:
            state = ep_data.iloc[step_idx][self._state_key]
            sample["observation.state"] = torch.tensor(np.array(state, dtype=np.float32))

        # Action chunk
        ep_len = len(ep_data)
        actions = []
        action_is_pad = []
        for t in range(self.action_horizon):
            future_idx = step_idx + t
            if future_idx < ep_len and "action" in ep_data.columns:
                act = np.array(ep_data.iloc[future_idx]["action"], dtype=np.float32)
                actions.append(act)
                action_is_pad.append(False)
            else:
                actions.append(actions[-1] if actions else np.zeros(7, dtype=np.float32))
                action_is_pad.append(True)

        sample["action"] = torch.tensor(np.stack(actions))
        sample["action_is_pad"] = torch.tensor(action_is_pad)

        # Task
        task_idx_col = "task_index" if "task_index" in ep_data.columns else None
        if task_idx_col:
            task_index = int(ep_data.iloc[step_idx][task_idx_col])
            sample["task"] = self._tasks.get(task_index, "")
            sample["task_index"] = torch.tensor(task_index, dtype=torch.int64)
        else:
            sample["task"] = self._tasks.get(0, "")
            sample["task_index"] = torch.tensor(0, dtype=torch.int64)

        # Metadata
        sample["episode_index"] = torch.tensor(episode_index, dtype=torch.int64)
        sample["frame_index"] = torch.tensor(step_idx, dtype=torch.int64)
        sample["index"] = torch.tensor(index, dtype=torch.int64)
        sample["timestamp"] = torch.tensor(step_idx / self.fps, dtype=torch.float32)

        if self._transform is not None:
            sample = self._transform(sample)

        return sample


class StreamingLeRobotV3Dataset(StreamingLeRobotDataset):
    """Decoupled VLA dataset built on top of StreamingLeRobotDataset (Iterable).

    Same decoupled interface as LeRobotV3Dataset but for streaming mode.
    Useful for large datasets that don't fit in memory.

    Args:
        repo_id: HuggingFace repo_id or local dataset identifier
        root: Path to local LeRobot v3.0 dataset root directory
        action_horizon: Number of future action steps (maps to chunk_size in PI05Config)
        episodes: Specific episode indices to load (None = all)
        image_transforms: Transform applied to visual modalities
        revision: Git revision (branch/tag/commit hash)
        tolerance_s: Timestamp tolerance for sync checks
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
        max_num_shards: int = 16,
        buffer_size: int = 1000,
        seed: int = 42,
        shuffle: bool = True,
        transform: Callable | None = None,
    ):
        self._action_horizon = action_horizon
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

        logger.info(
            f"StreamingLeRobotV3Dataset: repo_id={repo_id}, "
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
    buffer_size: int = 1000,
    seed: int = 42,
    shuffle: bool = True,
    lerobotdataset_version: str = "v3.0",
) -> Dataset:
    """Factory function to create VLA dataset with version dispatch.

    Args:
        lerobotdataset_version: Dataset format version ("v2.0", "v2.1", "v3.0").
            v2.0/v2.1 use JSONL metadata + one-parquet-per-episode (no lerobot lib needed).
            v3.0 uses lerobot official LeRobotDataset API.
        (other training_args: see individual dataset classes)

    Returns:
        Dataset instance matching the requested version/streaming mode.
    """
    if lerobotdataset_version in ("v2.0", "v2.1"):
        dataset = LeRobotV2Dataset(
            root=root,
            action_horizon=action_horizon,
            episodes=episodes,
            video_backend=video_backend,
        )
    elif lerobotdataset_version == "v3.0":
        if streaming:
            dataset = StreamingLeRobotV3Dataset(
                repo_id=repo_id,
                root=root,
                action_horizon=action_horizon,
                episodes=episodes,
                image_transforms=image_transforms,
                revision=revision,
                tolerance_s=tolerance_s,
                buffer_size=buffer_size,
                seed=seed,
                shuffle=shuffle,
            )
        else:
            dataset = LeRobotV3Dataset(
                repo_id=repo_id,
                root=root,
                action_horizon=action_horizon,
                episodes=episodes,
                image_transforms=image_transforms,
                revision=revision,
                video_backend=video_backend,
                tolerance_s=tolerance_s,
            )
    else:
        raise ValueError(
            f"Unsupported lerobotdataset_version: '{lerobotdataset_version}'. "
            f"Supported: v2.0, v2.1, v3.0"
        )

    return dataset


def build_lerobot_dataset(model_cfg, data_cfg, training_args):
    """Build lerobot-based VLA dataset from typed configs and CLI training_args."""
    dataset_path = training_args.dataset_path
    if not dataset_path:
        raise ValueError("Must specify --dataset-path")

    dataset_path = Path(dataset_path)
    repo_id = dataset_path.name

    action_horizon = model_cfg.action_horizon
    lerobotdataset_version = training_args.lerobotdataset_version
    video_backend = training_args.video_backend
    streaming = training_args.streaming

    return _build_lerobot_dataset(
        repo_id=repo_id,
        root=str(dataset_path),
        action_horizon=action_horizon,
        streaming=streaming,
        episodes=None,
        video_backend=video_backend,
        tolerance_s=1e-4,
        lerobotdataset_version=lerobotdataset_version,
    )
