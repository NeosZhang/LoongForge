"""
LeRobot Dataset Adapter - Load LeRobot v2.0/v3.0 format datasets

LeRobot is the robot data standard format led by HuggingFace, supporting:
  - LIBERO (4 suites, 40 tasks)
  - Bridge V2 (multi-scene tabletop manipulation)
  - RT-1 (Google real robot data)
  - Open X-Embodiment (70+ datasets in unified format)
  - RobotWin (simulation multi-task)
  - GR-1 (humanoid robot)

Data format:
  dataset_root/
    meta/
      modality.json       # Modality configuration
      episodes.jsonl      # Episode index
      tasks.jsonl         # Task descriptions
      info.json           # Dataset information
      stats_gr00t.json    # Statistics (for normalization)
    data/
      chunk-000/
        episode_000000.parquet   # Low-dimensional data (state, action)
    videos/
      chunk-000/
        episode_000000/
          video.primary_image.mp4

Usage (from config):
    datasets:
      vla_data:
        dataloader_module: lerobot_datasets
        data_root_dir: /data/lerobot
        dataset_mix: libero_spatial
        per_device_batch_size: 4
        action_horizon: 7
        image_size: 224
        normalization_mode: q99

Usage (programmatic):
    from dataloader.datasets.lerobot_dataset import LeRobotVLADataset

    dataset = LeRobotVLADataset(
        dataset_path="/data/lerobot/libero_spatial",
        action_horizon=7,
        image_size=224,
    )
    sample = dataset[0]
    # sample = {"image": [PIL.Image], "lang": "...", "action": ndarray[7,7], "state": ndarray[1,7]}
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataloader import collate_fn, save_dataset_statistics
from dataloader.transforms import ActionTransform, ComposedTransform, ImageTransform

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Classic open-source dataset Mixture registry
# ═══════════════════════════════════════════════════════════════

DATASET_NAMED_MIXTURES = {
    # ─── LIBERO Benchmark (4 suites) ─────────────────────
    "libero_spatial": [
        ("libero/libero_spatial", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero/libero_object", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero/libero_goal", 1.0, "libero_franka"),
    ],
    "libero_long": [
        ("libero/libero_long", 1.0, "libero_franka"),
    ],
    "libero_all": [
        ("libero/libero_spatial", 1.0, "libero_franka"),
        ("libero/libero_object", 1.0, "libero_franka"),
        ("libero/libero_goal", 1.0, "libero_franka"),
        ("libero/libero_long", 1.0, "libero_franka"),
    ],

    # ─── Bridge V2 (Berkeley) ────────────────────────────
    "bridge_v2": [
        ("bridge_v2/bridge_data_v2", 1.0, "oxe_bridge"),
    ],

    # ─── RT-1 (Google) ───────────────────────────────────
    "rt1": [
        ("rt1/fractal20220817_data", 1.0, "oxe_rt1"),
    ],

    # ─── Open X-Embodiment Mix ───────────────────────────
    "oxe_magic_soup": [
        ("bridge_v2/bridge_data_v2", 3.0, "oxe_bridge"),
        ("rt1/fractal20220817_data", 1.0, "oxe_rt1"),
        ("kuka/kuka", 0.5, "oxe_rt1"),
    ],

    # ─── Bimanual ─────────────────────────────────
    "aloha": [
        ("aloha/aloha_sim_transfer_cube", 1.0, "aloha"),
    ],

    # ─── Droid (multi-institution large-scale) ────────────────
    "droid": [
        ("droid/droid_100", 1.0, "oxe_droid"),
    ],
}


# ═══════════════════════════════════════════════════════════════
# Robot Type configuration (determines video/state/action key mapping)
# ═══════════════════════════════════════════════════════════════

ROBOT_TYPE_CONFIGS = {
    "libero_franka": {
        "video_keys": ["video.primary_image", "video.wrist_image"],
        "state_keys": ["state.eef_pos", "state.eef_quat", "state.gripper_pos"],
        "action_keys": ["action.eef_pos", "action.eef_quat", "action.gripper"],
        "action_dim": 7,
        "state_dim": 7,
        "normalization_mode": "q99",
        "gripper_indices": [6],
    },
    "oxe_bridge": {
        "video_keys": ["video.image_0"],
        "state_keys": ["state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw", "state.gripper"],
        "action_keys": ["action.x", "action.y", "action.z", "action.roll",
                        "action.pitch", "action.yaw", "action.gripper"],
        "action_dim": 7,
        "state_dim": 7,
        "normalization_mode": "q99",
        "gripper_indices": [6],
    },
    "oxe_rt1": {
        "video_keys": ["video.image"],
        "state_keys": [],
        "action_keys": ["action.world_vector", "action.rotation_delta", "action.gripper"],
        "action_dim": 7,
        "state_dim": 0,
        "normalization_mode": "q99",
        "gripper_indices": [6],
    },
    "oxe_droid": {
        "video_keys": ["video.exterior_image_1", "video.exterior_image_2", "video.wrist_image"],
        "state_keys": ["state.eef_position", "state.eef_rotation", "state.gripper_position"],
        "action_keys": ["action.eef_position_delta", "action.eef_rotation_delta", "action.gripper_position"],
        "action_dim": 7,
        "state_dim": 7,
        "normalization_mode": "min_max",
        "gripper_indices": [6],
    },
    "aloha": {
        "video_keys": ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"],
        "state_keys": ["state.left_arm", "state.right_arm"],
        "action_keys": ["action.left_arm", "action.right_arm"],
        "action_dim": 14,
        "state_dim": 14,
        "normalization_mode": "min_max",
        "gripper_indices": [6, 13],
    },
}


# ═══════════════════════════════════════════════════════════════
# LeRobot VLA Dataset
# ═══════════════════════════════════════════════════════════════

class LeRobotVLADataset(Dataset):
    """
    General LeRobot v2.0 dataset.

    Automatically reads episode data from parquet files,
    and reads task descriptions and statistics from meta/ directory.
    """

    def __init__(
        self,
        dataset_path: str,
        robot_type: str = "libero_franka",
        action_horizon: int = 7,
        image_size: int = 224,
        normalization_mode: Optional[str] = None,
        include_state: bool = True,
        video_backend: str = "pil",
    ):
        self.dataset_path = Path(dataset_path)
        self.robot_type = robot_type
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.include_state = include_state
        self.video_backend = video_backend

        # Robot config
        self.robot_cfg = ROBOT_TYPE_CONFIGS.get(robot_type, ROBOT_TYPE_CONFIGS["libero_franka"])
        self.normalization_mode = normalization_mode or self.robot_cfg["normalization_mode"]

        # Load metadata
        self._load_metadata()

        # Load data
        self._load_data()

        # Build transforms
        self.transform = self._build_transform()

        logger.info(
            f"LeRobotVLADataset: {self.dataset_path.name}, "
            f"{len(self)} samples, robot={robot_type}, "
            f"action_dim={self.robot_cfg['action_dim']}, horizon={action_horizon}"
        )

    def _load_metadata(self):
        """Load metadata from meta directory."""
        meta_dir = self.dataset_path / "meta"

        # Tasks (language instructions)
        self.tasks = {}
        tasks_file = meta_dir / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    task = json.loads(line)
                    self.tasks[task["task_index"]] = task["task"]

        # Statistics
        self.statistics = {}
        stats_file = meta_dir / "stats_gr00t.json"
        if not stats_file.exists():
            stats_file = meta_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self.statistics = json.load(f)

        # Episodes
        self.episodes = []
        episodes_file = meta_dir / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file) as f:
                for line in f:
                    self.episodes.append(json.loads(line))

    def _load_data(self):
        """Build a lazy frame index without loading parquet data into memory."""
        import pandas as pd

        data_dir = self.dataset_path / "data"
        parquet_files = sorted(data_dir.rglob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # Build index: list of (file_path, row_index_within_file)
        self._parquet_files = parquet_files
        self._frame_index: list[tuple[int, int]] = []  # (file_idx, row_in_file)
        self._file_cache: dict[int, pd.DataFrame] = {}  # LRU-like single-file cache
        self._cached_file_idx: int = -1

        for file_idx, pf in enumerate(parquet_files):
            # Read only row count via parquet metadata (no data loaded)
            import pyarrow.parquet as pq
            n_rows = pq.read_metadata(pf).num_rows
            self._frame_index.extend((file_idx, row) for row in range(n_rows))

        logger.info(f"Indexed {len(self._frame_index)} frames from {len(parquet_files)} parquet files (lazy load)")

    def _build_transform(self) -> ComposedTransform:
        """Build data transformation pipeline."""
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

        # Action transform
        action_stats = self._get_action_statistics()
        if action_stats:
            transforms.append(
                ActionTransform(
                    apply_to=["action"],
                    action_horizon=self.action_horizon,
                    normalization_mode=self.normalization_mode,
                    statistics=action_stats,
                    gripper_indices=self.robot_cfg.get("gripper_indices", []),
                )
            )
        else:
            transforms.append(
                ActionTransform(
                    apply_to=["action"],
                    action_horizon=self.action_horizon,
                    normalization_mode=self.normalization_mode,
                    statistics=None,
                )
            )

        return ComposedTransform(transforms)

    def _get_action_statistics(self) -> Optional[Dict]:
        """Extract action statistics from stats."""
        if not self.statistics:
            return None

        # Try different key patterns
        for key in ["action", "action.eef_pos", "actions"]:
            if key in self.statistics:
                return self.statistics[key]

        # Combine all action.* keys
        action_stats = {}
        for key, val in self.statistics.items():
            if key.startswith("action"):
                action_stats = val
                break

        return action_stats if action_stats else None

    def __len__(self) -> int:
        return len(self._frame_index)

    def _get_row(self, idx: int):
        """Fetch a single row, caching the current file to avoid re-reads."""
        import pandas as pd
        file_idx, row_in_file = self._frame_index[idx]
        if self._cached_file_idx != file_idx:
            self._file_cache = {}
            self._file_cache[file_idx] = pd.read_parquet(self._parquet_files[file_idx])
            self._cached_file_idx = file_idx
        return self._file_cache[file_idx].iloc[row_in_file]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._get_row(idx)

        # Extract images
        images = self._extract_images(row, idx)

        # Extract action
        action = self._extract_action(row, idx)

        # Extract state
        state = self._extract_state(row) if self.include_state else None

        # Extract language instruction
        lang = self._extract_language(row)

        sample = {
            "image": images,
            "lang": lang,
            "action": action,
            "state": state,
        }

        # Apply transforms
        sample = self.transform(sample)
        return sample

    def _extract_images(self, row, idx: int) -> List[Image.Image]:
        """Extract observation images."""
        images = []
        for vkey in self.robot_cfg["video_keys"]:
            col_name = vkey.replace("video.", "")
            if col_name in row.index:
                img_data = row[col_name]
                if isinstance(img_data, bytes):
                    import io
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                elif isinstance(img_data, np.ndarray):
                    img = Image.fromarray(img_data).convert("RGB")
                elif isinstance(img_data, str) and os.path.exists(img_data):
                    img = Image.open(img_data).convert("RGB")
                else:
                    # Placeholder
                    img = Image.new("RGB", (self.image_size, self.image_size))
                images.append(img)

        # Fallback: try loading from video files
        if not images:
            images = [Image.new("RGB", (self.image_size, self.image_size))]

        return images

    def _extract_action(self, row, idx: int) -> np.ndarray:
        """Extract action chunk."""
        action_dim = self.robot_cfg["action_dim"]
        actions = []

        # Try to get action from columns
        for akey in self.robot_cfg["action_keys"]:
            col_name = akey.replace("action.", "")
            if col_name in row.index:
                val = row[col_name]
                if isinstance(val, (list, np.ndarray)):
                    actions.extend(np.array(val).flatten().tolist())
                elif isinstance(val, (int, float)):
                    actions.append(float(val))

        if not actions:
            # Fallback: try concatenated "action" column
            if "action" in row.index:
                val = row["action"]
                if isinstance(val, (list, np.ndarray)):
                    actions = np.array(val).flatten().tolist()

        if not actions:
            actions = [0.0] * action_dim

        # Get action chunk from consecutive frames
        action_array = np.array(actions[:action_dim], dtype=np.float32)

        # For now, return single-step action (will be chunked by ActionTransform)
        return action_array.reshape(1, -1)

    def _extract_state(self, row) -> Optional[np.ndarray]:
        """Extract state."""
        state_dim = self.robot_cfg["state_dim"]
        if state_dim == 0:
            return None

        states = []
        for skey in self.robot_cfg["state_keys"]:
            col_name = skey.replace("state.", "")
            if col_name in row.index:
                val = row[col_name]
                if isinstance(val, (list, np.ndarray)):
                    states.extend(np.array(val).flatten().tolist())
                elif isinstance(val, (int, float)):
                    states.append(float(val))

        if not states:
            return None

        return np.array(states[:state_dim], dtype=np.float32).reshape(1, -1)

    def _extract_language(self, row) -> str:
        """Extract language instruction."""
        # Try common column names
        for col in ["task", "language_instruction", "lang", "instruction", "task_description"]:
            if col in row.index:
                val = row[col]
                if isinstance(val, str) and val:
                    return val

        # Try task_index -> tasks mapping
        if "task_index" in row.index:
            task_idx = int(row["task_index"])
            if task_idx in self.tasks:
                return self.tasks[task_idx]

        return "perform the task"

    def get_dataset_statistics(self) -> Dict:
        """Return dataset statistics (for action denormalization during inference)."""
        return self.statistics


# ═══════════════════════════════════════════════════════════════
# Mixture Dataset (multi-dataset weighted mixing)
# ═══════════════════════════════════════════════════════════════

class LeRobotMixtureDataset(Dataset):
    """
    Multi-dataset weighted mixture sampling.

    Samples from sub-datasets according to weight proportions.
    """

    def __init__(
        self,
        datasets: List[Tuple[Dataset, float]],
        seed: int = 42,
    ):
        self.datasets = datasets
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Compute sampling probabilities
        total_weight = sum(w for _, w in datasets)
        self.probs = [w / total_weight for _, w in datasets]
        self.cum_probs = np.cumsum(self.probs)

        # Total length = sum of all dataset lengths
        self._total_len = sum(len(d) for d, _ in datasets)

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Weighted random selection
        r = self._rng.random()
        dataset_idx = np.searchsorted(self.cum_probs, r)
        dataset_idx = min(dataset_idx, len(self.datasets) - 1)

        ds, _ = self.datasets[dataset_idx]
        sample_idx = self._rng.randint(0, len(ds))
        return ds[sample_idx]


# ═══════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════

def build_lerobot_dataloader(cfg, vla_cfg, batch_size: int, num_workers: int) -> DataLoader:
    """
    Build LeRobot dataloader.

    Supports:
      - Single dataset: specify dataset_path + robot_type
      - Mixture: specify dataset_mix name (looked up from DATASET_NAMED_MIXTURES)
    """
    data_root_dir = vla_cfg.get("data_root_dir", "/data/lerobot")
    dataset_mix = vla_cfg.get("dataset_mix", None)
    dataset_path = vla_cfg.get("dataset_path", None)
    robot_type = vla_cfg.get("robot_type", "libero_franka")
    action_horizon = vla_cfg.get("action_horizon", 7)
    image_size = vla_cfg.get("image_size", 224)
    normalization_mode = vla_cfg.get("normalization_mode", None)

    if dataset_mix and dataset_mix in DATASET_NAMED_MIXTURES:
        # Mixture mode
        mixture_spec = DATASET_NAMED_MIXTURES[dataset_mix]
        datasets = []
        for d_name, d_weight, d_robot_type in mixture_spec:
            d_path = os.path.join(data_root_dir, d_name)
            if not os.path.exists(d_path):
                logger.warning(f"Dataset path not found: {d_path}, skipping")
                continue
            ds = LeRobotVLADataset(
                dataset_path=d_path,
                robot_type=d_robot_type,
                action_horizon=action_horizon,
                image_size=image_size,
                normalization_mode=normalization_mode,
            )
            datasets.append((ds, d_weight))

        if not datasets:
            raise FileNotFoundError(
                f"No valid datasets found for mixture '{dataset_mix}' under {data_root_dir}"
            )

        dataset = LeRobotMixtureDataset(datasets)
    elif dataset_path:
        # Single dataset mode
        dataset = LeRobotVLADataset(
            dataset_path=dataset_path,
            robot_type=robot_type,
            action_horizon=action_horizon,
            image_size=image_size,
            normalization_mode=normalization_mode,
        )
    else:
        raise ValueError(
            "Must specify either 'dataset_mix' or 'dataset_path' in config"
        )

    # Save statistics
    output_dir = getattr(cfg, "output_dir", None)
    if output_dir and hasattr(dataset, "get_dataset_statistics"):
        stats = dataset.get_dataset_statistics()
        if stats:
            save_dataset_statistics(stats, os.path.join(output_dir, "dataset_statistics.json"))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
