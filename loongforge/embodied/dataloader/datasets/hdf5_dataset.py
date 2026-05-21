"""
HDF5 Dataset Adapter - Load HDF5 format datasets

Supports classic open-source datasets:
  - RoboMimic (robomimic): NutAssembly, Can, Lift, Square, etc.
  - RoboCasa: Kitchen manipulation (365 tasks)
  - LIBERO (HDF5 original format): libero_spatial, libero_object, libero_goal, libero_long

HDF5 data format (robomimic style):
  demo.hdf5
    data/
      demo_0/
        obs/
          agentview_rgb: (T, H, W, 3) uint8
          eye_in_hand_rgb: (T, H, W, 3) uint8
          robot0_eef_pos: (T, 3) float64
          robot0_eef_quat: (T, 4) float64
          robot0_gripper_qpos: (T, 2) float64
        actions: (T, 7) float64
      demo_1/ ...
    mask/
      train: [demo_0, demo_1, ...]
      valid: [demo_10, ...]

Usage (from config):
    datasets:
      vla_data:
        dataloader_module: hdf5_datasets
        data_dir: /data/robomimic/can/ph/low_dim_v141.hdf5
        task_name: "pick up the can"
        per_device_batch_size: 8
        action_horizon: 10
        image_size: 224

Usage (programmatic):
    from dataloader.datasets.hdf5_dataset import HDF5VLADataset

    dataset = HDF5VLADataset(
        hdf5_path="/data/robomimic/can/ph/image_v141.hdf5",
        task_name="pick up the can and place it in the bin",
        action_horizon=10,
        image_size=224,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
    )
    sample = dataset[0]
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataloader import collate_fn
from dataloader.transforms import ActionTransform, ComposedTransform, ImageTransform

logger = logging.getLogger(__name__)


class HDF5VLADataset(Dataset):
    """
    HDF5 format VLA dataset (compatible with RoboMimic/LIBERO/RoboCasa).

    Each demo episode is expanded into multiple (obs_t, action_chunk) samples.
    """

    def __init__(
        self,
        hdf5_path: str,
        task_name: str = "perform the task",
        action_horizon: int = 10,
        image_size: int = 224,
        image_keys: Optional[List[str]] = None,
        state_keys: Optional[List[str]] = None,
        action_key: str = "actions",
        obs_prefix: str = "obs",
        split: str = "train",
        normalization_mode: str = "min_max",
    ):
        """
        Args:
            hdf5_path: HDF5 file path
            task_name: Language description of the task
            action_horizon: Action chunk length
            image_size: Image resize size
            image_keys: Image observation key list
            state_keys: Low-dimensional state key list
            action_key: Action data key name
            obs_prefix: Observation group prefix
            split: Dataset split to use (train/valid)
            normalization_mode: Action normalization mode
        """
        import h5py

        self.hdf5_path = hdf5_path
        self.task_name = task_name
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.image_keys = image_keys or ["agentview_rgb", "eye_in_hand_rgb"]
        self.state_keys = state_keys or ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        self.action_key = action_key
        self.obs_prefix = obs_prefix
        self.normalization_mode = normalization_mode

        # Open HDF5 and build index
        self._h5 = h5py.File(hdf5_path, "r")
        self._demos = self._get_demos(split)
        self._index = self._build_index()

        # Compute statistics
        self._statistics = self._compute_statistics()

        # Build transform
        self.transform = self._build_transform()

        logger.info(
            f"HDF5VLADataset: {Path(hdf5_path).name}, "
            f"{len(self._demos)} demos, {len(self)} samples, "
            f"action_horizon={action_horizon}"
        )

    def _get_demos(self, split: str) -> List[str]:
        """Get demo list for specified split."""
        if "mask" in self._h5 and split in self._h5["mask"]:
            demo_names = [
                name.decode() if isinstance(name, bytes) else name
                for name in self._h5["mask"][split][:]
            ]
        else:
            # Fallback: use all demos
            demo_names = sorted(self._h5["data"].keys())
        return demo_names

    def _build_index(self) -> List[Tuple[str, int]]:
        """Build (demo_name, timestep) index."""
        index = []
        for demo_name in self._demos:
            demo = self._h5["data"][demo_name]
            T = demo[self.action_key].shape[0]
            for t in range(T):
                index.append((demo_name, t))
        return index

    def _compute_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute action statistics."""
        all_actions = []
        for demo_name in self._demos:
            demo = self._h5["data"][demo_name]
            actions = demo[self.action_key][:]
            all_actions.append(actions)

        all_actions = np.concatenate(all_actions, axis=0)
        stats = {
            "action": {
                "mean": np.mean(all_actions, axis=0),
                "std": np.std(all_actions, axis=0),
                "min": np.min(all_actions, axis=0),
                "max": np.max(all_actions, axis=0),
                "q01": np.percentile(all_actions, 1, axis=0),
                "q99": np.percentile(all_actions, 99, axis=0),
            }
        }
        return stats

    def _build_transform(self) -> ComposedTransform:
        transforms = [
            ImageTransform(
                apply_to=["image"],
                size=(self.image_size, self.image_size),
                crop_scale=0.95,
                color_jitter=True,
            ),
            ActionTransform(
                apply_to=["action"],
                action_horizon=self.action_horizon,
                normalization_mode=self.normalization_mode,
                statistics=self._statistics.get("action"),
            ),
        ]
        return ComposedTransform(transforms)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        demo_name, t = self._index[idx]
        demo = self._h5["data"][demo_name]
        obs = demo[self.obs_prefix]

        # Images
        images = []
        for img_key in self.image_keys:
            if img_key in obs:
                img_data = obs[img_key][t]  # (H, W, 3)
                img = Image.fromarray(img_data.astype(np.uint8)).convert("RGB")
                images.append(img)
        if not images:
            images = [Image.new("RGB", (self.image_size, self.image_size))]

        # Action chunk: [t, t+action_horizon)
        actions = demo[self.action_key][:]
        T = actions.shape[0]
        action_chunk = actions[t: t + self.action_horizon]
        if len(action_chunk) < self.action_horizon:
            # Pad with last action
            pad_len = self.action_horizon - len(action_chunk)
            padding = np.tile(action_chunk[-1:], (pad_len, 1))
            action_chunk = np.concatenate([action_chunk, padding], axis=0)

        # State
        state_parts = []
        for skey in self.state_keys:
            if skey in obs:
                state_parts.append(obs[skey][t].flatten())
        state = np.concatenate(state_parts).astype(np.float32) if state_parts else None

        # Language
        lang = self.task_name
        # Some LIBERO HDF5 files store task in attrs
        if "language_instruction" in demo.attrs:
            lang = demo.attrs["language_instruction"]
            if isinstance(lang, bytes):
                lang = lang.decode()

        sample = {
            "image": images,
            "lang": lang,
            "action": action_chunk.astype(np.float32),
            "state": state.reshape(1, -1) if state is not None else None,
        }

        sample = self.transform(sample)
        return sample

    def get_dataset_statistics(self) -> Dict:
        """Return dataset statistics."""
        return self._statistics

    def __del__(self):
        if hasattr(self, "_h5") and self._h5:
            self._h5.close()


def build_hdf5_dataloader(cfg, vla_cfg, batch_size: int, num_workers: int) -> DataLoader:
    """Build HDF5 dataloader."""
    data_dir = vla_cfg.get("data_dir", "")
    task_name = vla_cfg.get("task_name", "perform the task")
    action_horizon = vla_cfg.get("action_horizon", 10)
    image_size = vla_cfg.get("image_size", 224)
    image_keys = vla_cfg.get("image_keys", None)
    state_keys = vla_cfg.get("state_keys", None)
    normalization_mode = vla_cfg.get("normalization_mode", "min_max")

    # Support both single file and directory of files
    if os.path.isfile(data_dir) and data_dir.endswith(".hdf5"):
        hdf5_paths = [data_dir]
    elif os.path.isdir(data_dir):
        hdf5_paths = sorted(Path(data_dir).rglob("*.hdf5"))
    else:
        raise FileNotFoundError(f"HDF5 data not found: {data_dir}")

    # For simplicity, load first file (extend to multi-file as needed)
    dataset = HDF5VLADataset(
        hdf5_path=str(hdf5_paths[0]),
        task_name=task_name,
        action_horizon=action_horizon,
        image_size=image_size,
        image_keys=image_keys,
        state_keys=state_keys,
        normalization_mode=normalization_mode,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
