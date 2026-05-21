"""
Normalizer - Multi-mode normalizer

Supports 4 normalization modes:
  - q99:     2*(x - q01) / (q99 - q01) - 1  → [-1, 1]
  - min_max: 2*(x - min) / (max - min) - 1  → [-1, 1]
  - mean_std: (x - mean) / std              → unbounded
  - binary:  x > threshold                  → {0, 1}

Statistics dict format:
    {"mean": [...], "std": [...], "min": [...], "max": [...], "q01": [...], "q99": [...]}
"""

from typing import Dict

import numpy as np
import torch


class Normalizer:
    """General normalizer, supports forward/inverse."""

    VALID_MODES = ["q99", "min_max", "mean_std", "binary"]

    def __init__(self, mode: str, statistics: Dict[str, np.ndarray], binary_threshold: float = 0.5):
        """
        Args:
            mode: Normalization mode (q99, min_max, mean_std, binary)
            statistics: Dataset statistics dictionary
            binary_threshold: Threshold for binary mode
        """
        assert mode in self.VALID_MODES, f"Invalid mode: {mode}. Valid: {self.VALID_MODES}"
        self.mode = mode
        self.binary_threshold = binary_threshold

        # Convert to tensors
        self.statistics = {}
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                self.statistics[key] = torch.from_numpy(value).float()
            elif isinstance(value, (list, tuple)):
                self.statistics[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                self.statistics[key] = value.float()
            else:
                self.statistics[key] = torch.tensor(value, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            mask = q01 != q99
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (
                2 * (x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask]) - 1
            )
            normalized[..., ~mask] = x[..., ~mask]
            return torch.clamp(normalized, -1, 1)

        elif self.mode == "min_max":
            mn = self.statistics["min"].to(x.dtype)
            mx = self.statistics["max"].to(x.dtype)
            mask = mn != mx
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (
                2 * (x[..., mask] - mn[..., mask]) / (mx[..., mask] - mn[..., mask]) - 1
            )
            normalized[..., ~mask] = 0
            return normalized

        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            mask = std != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]
            normalized[..., ~mask] = x[..., ~mask]
            return normalized

        elif self.mode == "binary":
            return (x > self.binary_threshold).to(x.dtype)

        raise ValueError(f"Invalid mode: {self.mode}")

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize (used during inference)."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x + 1) / 2 * (q99 - q01) + q01

        elif self.mode == "min_max":
            mn = self.statistics["min"].to(x.dtype)
            mx = self.statistics["max"].to(x.dtype)
            return (x + 1) / 2 * (mx - mn) + mn

        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean

        elif self.mode == "binary":
            return (x > self.binary_threshold).to(x.dtype)

        raise ValueError(f"Invalid mode: {self.mode}")

    def __call__(self, x, inverse=False):
        return self.inverse(x) if inverse else self.forward(x)
