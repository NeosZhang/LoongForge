"""
测试 QwenFast 数据加载；
为了对齐 starvla，读取 LeRobot V2 格式数据
不绑定特定模型架构；
开发中或许常用，后续版本删掉/合并

Usage:
    from dataloader.datasets.lerobot_v2_spatial import build_lerobot_v2_dataloader
    dataloader = build_lerobot_v2_dataloader(cfg)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pyarrow.parquet as pq

from loongforge.embodied.data.transforms import (
    ActionTransform,
    ComposedTransform,
    ImageTransform,
)

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Default collate: directly return batch list."""
    return batch


class LeRobotV2SpatialDataset(Dataset):
    """Simple LeRobot V2 dataset loader for spatial manipulation tasks."""

    def __init__(
        self,
        dataset_path: str | Path,
        action_horizon: int = 8,
        action_dim: int = 7,
        image_size: int = 224,
        normalization_mode: Optional[str] = "q99",
    ):
        self.dataset_path = Path(dataset_path)
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.image_size = image_size
        self.normalization_mode = normalization_mode

        self._load_metadata()
        self._build_index()
        self.transform = self._build_transform()

        logger.info(f"Loaded LeRobotV2: {len(self._frame_index)} frames from {len(self._parquet_files)} files")

    def _load_metadata(self):
        """Load dataset metadata."""
        meta_dir = self.dataset_path / "meta"

        # Load statistics - handle stats_gr00t.json format
        self.statistics = {}
        stats_file = meta_dir / "stats_gr00t.json"
        if stats_file.exists():
            with open(stats_file) as f:
                raw = json.load(f)
            if "statistics" in raw:
                self.statistics = raw["statistics"]
            else:
                self.statistics = raw

        # Load episodes
        self.episodes = []
        episodes_file = meta_dir / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file) as f:
                for line in f:
                    self.episodes.append(json.loads(line))

        # Load tasks
        self.tasks = {}
        tasks_file = meta_dir / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    task = json.loads(line)
                    self.tasks[task["task_index"]] = task.get("task", "")

    def _build_index(self):
        """Build frame index from parquet files."""
        data_dir = self.dataset_path / "data"
        all_files = sorted(data_dir.rglob("*.parquet"))
        self._parquet_files = [f for f in all_files if not f.name.startswith("._")]

        if not self._parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        self._frame_index = []
        self._cached_df = None
        self._cached_file_idx = -1

        for file_idx, pf in enumerate(self._parquet_files):
            n_rows = pq.read_metadata(str(pf)).num_rows
            self._frame_index.extend((file_idx, row) for row in range(n_rows))

    def _build_transform(self) -> ComposedTransform:
        """Build transform pipeline with ActionTransform for action horizon."""
        transforms = []

        # Image transform
        transforms.append(
            ImageTransform(
                apply_to=["image"],
                size=(self.image_size, self.image_size),
                crop_scale=0.95,
                color_jitter=True,
            )
        )

        # Action transform - extends single-frame action to action_horizon
        action_stats = self._get_action_statistics()
        transforms.append(
            ActionTransform(
                apply_to=["action"],
                action_horizon=self.action_horizon,
                normalization_mode=self.normalization_mode,
                statistics=action_stats,
                gripper_indices=[6],  # libero_franka gripper index
            )
        )

        return ComposedTransform(transforms)

    def _get_action_statistics(self) -> Optional[Dict]:
        """Get action statistics for normalization."""
        if not hasattr(self, "statistics") or not self.statistics:
            return None

        for key in ["action", "action.eef_pos", "actions"]:
            if key in self.statistics:
                return self.statistics[key]

        for key, val in self.statistics.items():
            if key.startswith("action"):
                return val

        return None

    def _get_video_frame(self, episode_index: int, frame_index: int) -> Image.Image:
        """Load a single video frame."""
        video_dir = self.dataset_path / "videos" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}"

        video_file = video_dir / "video.primary_image.mp4"
        if video_file.exists():
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_file))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame)
            except Exception:
                pass

        images_dir = video_dir / "images"
        if images_dir.exists():
            image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
            if frame_index < len(image_files):
                return Image.open(image_files[frame_index])

        return Image.new("RGB", (self.image_size, self.image_size))

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, row_idx = self._frame_index[idx]

        if self._cached_file_idx != file_idx:
            self._cached_df = pd.read_parquet(str(self._parquet_files[file_idx]))
            self._cached_file_idx = file_idx

        row = self._cached_df.iloc[row_idx]

        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        image = self._get_video_frame(episode_index, frame_index)

        action = np.array(row["action"], dtype=np.float32)
        if action.ndim > 1:
            action = action[0] if len(action) > 0 else action
        # Reshape to (1, action_dim) for ActionTransform
        action = action.reshape(1, -1)

        task_index = int(row["task_index"])
        instruction = self.tasks.get(task_index, "")
        if not instruction and task_index < len(self.episodes):
            instruction = self.episodes[task_index].get("task", "")

        sample = {
            "image": [image],
            "lang": instruction,
            "action": action,
        }

        # Apply transform (ImageTransform + ActionTransform)
        sample = self.transform(sample)
        return sample

    def get_dataset_statistics(self) -> Dict:
        """Return the loaded dataset statistics (or an empty dict)."""
        return self.statistics


def build_mock_lerobot_v2_dataset(model_cfg, args) -> LeRobotV2SpatialDataset:
    """Build dataset for LeRobot V2 format.

    Args:
        model_cfg: OmegaConf model config (for transforms that need model info)
        args: CLI args namespace (contains dataset path, batch size, etc.)

    Returns:
        LeRobotV2SpatialDataset instance
    """
    dataset_path = getattr(args, "dataset_path", None)
    action_horizon = model_cfg.get("action_model", {}).get("action_horizon", 8)
    action_dim = model_cfg.get("action_model", {}).get("action_dim", 7)
    image_size = model_cfg.get("backbone", {}).get("image_size", 224)
    normalization_mode = getattr(args, "normalization_mode", "q99")

    if not dataset_path:
        raise ValueError("dataset_path must be specified via --dataset-path")

    dataset = LeRobotV2SpatialDataset(
        dataset_path=dataset_path,
        action_horizon=action_horizon,
        action_dim=action_dim,
        image_size=image_size,
        normalization_mode=normalization_mode,
    )

    return dataset
