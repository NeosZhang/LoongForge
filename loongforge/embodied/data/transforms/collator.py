# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Base collator framework for DataLoader collate functions.

Classes:
    - PreparedBatch: Base dataclass for model-ready batch tensors
    - BasePreprocessor: Abstract base for collate functions
    - register_preprocessor: Decorator to register model-specific collators
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Type

import torch


_PREPROCESSOR_REGISTRY: Dict[str, Type["BasePreprocessor"]] = {}


def register_preprocessor(name: str):
    """Decorator to register a preprocessor class for a model."""
    def decorator(cls):
        _PREPROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


@dataclass
class PreparedBatch:
    """Base class for preprocessed batch data.

    All tensor fields are on CPU after collation.
    Call .to(device) to move everything to GPU before forward().
    """
    def to(self, device: torch.device) -> "PreparedBatch":
        """Move all tensor fields to the given device. Returns self."""
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(device))
            elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                setattr(self, f.name, [t.to(device) for t in val])
        return self


class BasePreprocessor:
    """Abstract base for model-specific DataLoader collate functions."""

    @classmethod
    def from_config(cls, cfg) -> "BasePreprocessor":
        """Construct preprocessor from a full config object."""
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config(cfg) classmethod"
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> PreparedBatch:
        """Transform a list of dataset samples into a PreparedBatch."""
        raise NotImplementedError


def get_preprocessor(name: str) -> Type[BasePreprocessor]:
    """Look up a registered preprocessor class by name."""
    if name not in _PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"Unknown preprocessor '{name}'. "
            f"Available: {list(_PREPROCESSOR_REGISTRY.keys())}"
        )
    return _PREPROCESSOR_REGISTRY[name]


def build_preprocessor(name: str, cfg) -> BasePreprocessor:
    """Instantiate a registered preprocessor via its from_config classmethod."""
    cls = get_preprocessor(name)
    return cls.from_config(cfg)


@register_preprocessor("dummy")
class DummyPreprocessor(BasePreprocessor):
    """Pass-through preprocessor that returns examples as-is in a PreparedBatch."""

    @classmethod
    def from_config(cls, cfg) -> "DummyPreprocessor":
        return cls()

    def __call__(self, examples: List[Dict[str, Any]]) -> PreparedBatch:
        return PreparedBatch()
