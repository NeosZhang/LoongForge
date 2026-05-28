# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""XVLA batch-level collator.

The XVLA model consumes LeRobot-style dictionary batches directly. Keep the
original feature keys (``observation.*``, ``action``) instead of converting them
into a private dataclass, so the training path matches the reference LeRobot
batch layout.
"""

from typing import Any, Dict, List

import torch

from loongforge.embodied.data.transforms.collator import (
    BasePreprocessor,
    register_preprocessor,
)


class XVLABatch(dict):
    """Dictionary batch with a tensor-recursive ``to(device)`` helper."""

    def to(self, device: torch.device) -> "XVLABatch":
        """Recursively move all tensor values to the target device."""
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [move(v) for v in value]
            if isinstance(value, tuple):
                return tuple(move(v) for v in value)
            return value

        for key, value in list(self.items()):
            self[key] = move(value)
        return self


@register_preprocessor("xvla")
class XVLAPreprocessor(BasePreprocessor):
    """DataLoader collate_fn for XVLA.

    Mirrors LeRobot's default collation: tensors are stacked under their original
    keys, strings stay as Python lists, and metadata dictionaries are preserved.
    """

    @classmethod
    def from_config(cls, cfg) -> "XVLAPreprocessor":
        """Construct XVLAPreprocessor from a config object."""
        return cls()

    def __call__(self, examples: List[Dict[str, Any]]) -> XVLABatch:
        """Collate a list of sample dicts into a stacked XVLABatch."""
        keys = set().union(*(example.keys() for example in examples))
        batch = XVLABatch()
        for key in sorted(keys):
            values = [example.get(key) for example in examples]
            if all(isinstance(value, torch.Tensor) for value in values):
                batch[key] = torch.stack(values)
            elif all(isinstance(value, (int, bool)) for value in values):
                batch[key] = torch.tensor(values, dtype=torch.long)
            elif all(isinstance(value, float) for value in values):
                batch[key] = torch.tensor(values, dtype=torch.float32)
            else:
                batch[key] = values
        return batch
