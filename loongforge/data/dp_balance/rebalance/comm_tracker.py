# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Communication overhead tracker for DP balance operations."""

import numpy as np


class _CommOverheadTracker:
    """Tracks communication overhead for DP balance rebalancing operations.

    Collects actual timing data for redistribute_tensors operations during
    training and fits a model to estimate communication cost.
    """

    def __init__(self):
        self._measurements = []
        self._fitted = False
        self._coef_a = 0.0
        self._coef_b = 0.0

    def record(self, data_size: int, comm_time_ms: float):
        """Record a communication measurement.

        Args:
            data_size: Total number of elements redistributed.
            comm_time_ms: Measured communication time in milliseconds.
        """
        self._measurements.append((data_size, comm_time_ms))
        self._fitted = False

    def _fit(self):
        """Fit linear model: comm_time = a * data_size + b"""
        if len(self._measurements) < 2:
            return

        x = np.array([m[0] for m in self._measurements])
        y = np.array([m[1] for m in self._measurements])
        self._coef_a, self._coef_b = np.polyfit(x, y, 1)
        self._fitted = True

    def estimate(self, data_size: int) -> float:
        """Estimate communication time for given data size.

        Args:
            data_size: Total number of elements to redistribute.

        Returns:
            Estimated communication time in milliseconds.
        """
        if not self._fitted:
            self._fit()

        if self._fitted:
            return self._coef_a * data_size + self._coef_b
        else:
            if self._measurements:
                avg_ratio = sum(m[1] / (m[0] + 1e-6) for m in self._measurements) / len(self._measurements)
                return avg_ratio * data_size
            return 0.0


# Global tracker instance
_comm_overhead_tracker = _CommOverheadTracker()