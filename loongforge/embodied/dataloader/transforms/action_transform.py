"""
ActionTransform - Action/State preprocessing transform

Supports:
  - Action chunking (truncate/pad to fixed horizon)
  - Action normalization (q99 / min_max / mean_std / binary)
  - State normalization
  - Gripper binary processing
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from dataloader.transforms.base import BaseTransform
from dataloader.transforms.normalizer import Normalizer


class ActionTransform(BaseTransform):
    """
    Action/State normalization + chunking transform.

    Typical usage:
        transform = ActionTransform(
            apply_to=["action"],
            action_horizon=7,
            normalization_mode="q99",
            statistics={"q01": [...], "q99": [...]},
        )
    """

    def __init__(
        self,
        apply_to: List[str],
        action_horizon: int = 7,
        normalization_mode: str = "q99",
        statistics: Optional[Dict[str, Any]] = None,
        gripper_indices: Optional[List[int]] = None,
        gripper_binary_threshold: float = 0.5,
        training: bool = True,
    ):
        """
        Args:
            apply_to: List of keys to transform (e.g., ["action", "state"])
            action_horizon: Target action chunk length
            normalization_mode: Normalization mode
            statistics: Dataset statistics (None to skip normalization)
            gripper_indices: Gripper dimension indices (processed independently with binary mode)
            gripper_binary_threshold: Gripper binarization threshold
        """
        super().__init__(apply_to=apply_to, training=training)
        self.action_horizon = action_horizon
        self.normalization_mode = normalization_mode
        self.gripper_indices = gripper_indices or []
        self.gripper_binary_threshold = gripper_binary_threshold

        # Build normalizer
        self.normalizer = None
        if statistics is not None:
            self.normalizer = Normalizer(
                mode=normalization_mode,
                statistics=statistics,
            )

        # Gripper normalizer (binary)
        self.gripper_normalizer = None
        if self.gripper_indices and statistics is not None:
            # Extract gripper statistics
            gripper_stats = {}
            for key, val in statistics.items():
                if isinstance(val, (list, np.ndarray)):
                    arr = np.array(val)
                    if len(arr) > max(self.gripper_indices):
                        gripper_stats[key] = arr[self.gripper_indices]
            if gripper_stats:
                self.gripper_normalizer = Normalizer(
                    mode="binary",
                    statistics=gripper_stats,
                    binary_threshold=gripper_binary_threshold,
                )

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """apply for transform"""
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if value is None:
                continue

            # Convert to numpy
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            elif isinstance(value, list):
                value = np.array(value, dtype=np.float32)

            # Ensure 2D: [horizon, dim]
            if value.ndim == 1:
                value = value[np.newaxis, :]

            # Action chunking: pad/truncate to action_horizon
            if key == "action":
                value = self._chunk_action(value)

            # Normalize
            if self.normalizer is not None:
                value_tensor = torch.from_numpy(value).float()

                if self.gripper_indices and self.gripper_normalizer:
                    # Normalize non-gripper dims with main normalizer
                    non_gripper_mask = np.ones(value.shape[-1], dtype=bool)
                    non_gripper_mask[self.gripper_indices] = False

                    # Normalize all with main normalizer first
                    normalized = self.normalizer.forward(value_tensor)

                    # Override gripper dims with binary
                    gripper_values = value_tensor[..., self.gripper_indices]
                    normalized[..., self.gripper_indices] = (
                        gripper_values > self.gripper_binary_threshold
                    ).float()

                    value = normalized.numpy()
                else:
                    value = self.normalizer.forward(value_tensor).numpy()

            data[key] = value.astype(np.float32)
        return data

    def unapply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Denormalization (used during inference)."""
        for key in self.apply_to:
            if key not in data or data[key] is None:
                continue
            if self.normalizer is not None:
                value = data[key]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                data[key] = self.normalizer.inverse(value).numpy()
        return data

    def _chunk_action(self, action: np.ndarray) -> np.ndarray:
        """Truncate or pad action to action_horizon length."""
        T, D = action.shape
        if T >= self.action_horizon:
            return action[: self.action_horizon]
        else:
            # Pad with last action (repeat padding)
            pad_len = self.action_horizon - T
            padding = np.tile(action[-1:], (pad_len, 1))
            return np.concatenate([action, padding], axis=0)
