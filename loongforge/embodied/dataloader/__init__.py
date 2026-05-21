"""
LoongForgeVLA Dataloader - General data loading framework

Provides:
  - build_dataloader(): Unified factory function, dispatches to specific backend based on config
  - transforms/: General State/Action/Image transformation pipeline
  - datasets/: Open-source dataset adapters
    - lerobot_dataset: LeRobot v2.0 format (LIBERO, Bridge, RT-1, etc.)
    - rlds_dataset: RLDS/TFDS format (Open X-Embodiment)
    - hdf5_dataset: HDF5 format (RoboMimic, RoboCasa)

Architecture:
    cfg.datasets.vla_data.dataloader_module → selects backend
    Each backend returns a torch.utils.data.DataLoader

Sample output format (VLA dataset __getitem__):
    {
        "image": [PIL.Image, ...],          # Observation image list (primary + wrist, etc.)
        "lang": str,                        # Language instruction
        "action": np.ndarray,               # [action_horizon, action_dim] float32
        "state": np.ndarray | None,         # [state_horizon, state_dim] float32
    }
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Default collate: directly return batch list, batching handled internally by model."""
    return batch


def save_dataset_statistics(dataset_statistics: Dict, output_path):
    """Save dataset_statistics.json for action denormalization during inference."""
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
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


def build_dataloader(cfg, dataloader_module: Optional[str] = None) -> DataLoader:
    """
    Unified dataloader factory.

    Dispatches to the corresponding dataset implementation based on
    cfg.datasets.vla_data.dataloader_module.

    Supported backends:
      - "lerobot_datasets": LeRobot v2.0 format (recommended, compatible with LIBERO/Bridge/RT-1/OXE)
      - "rlds_datasets": RLDS/TFDS format (Open X-Embodiment)
      - "hdf5_datasets": HDF5 format (RoboMimic/RoboCasa/LIBERO-HDF5)
      - "dummy_datasets": Synthetic data (for debugging)

    Args:
        cfg: OmegaConf configuration, must contain cfg.datasets.vla_data
        dataloader_module: Override the module specified in config

    Returns:
        torch.utils.data.DataLoader
    """
    vla_cfg = cfg.datasets.vla_data
    module = dataloader_module or vla_cfg.get("dataloader_module", "lerobot_datasets")
    batch_size = vla_cfg.get("per_device_batch_size", 2)
    num_workers = vla_cfg.get("num_workers", 4)

    if module == "lerobot_datasets":
        from dataloader.datasets.lerobot_dataset import build_lerobot_dataloader

        return build_lerobot_dataloader(cfg, vla_cfg, batch_size, num_workers)

    elif module == "rlds_datasets":
        from dataloader.datasets.rlds_dataset import build_rlds_dataloader

        return build_rlds_dataloader(cfg, vla_cfg, batch_size, num_workers)

    elif module == "hdf5_datasets":
        from dataloader.datasets.hdf5_dataset import build_hdf5_dataloader

        return build_hdf5_dataloader(cfg, vla_cfg, batch_size, num_workers)

    elif module == "dummy_datasets":
        from dataloader.datasets.dummy_dataset import build_dummy_dataloader

        return build_dummy_dataloader(cfg, vla_cfg, batch_size, num_workers)

    else:
        raise ValueError(
            f"Unknown dataloader_module: '{module}'. "
            f"Supported: lerobot_datasets, rlds_datasets, hdf5_datasets, dummy_datasets"
        )
