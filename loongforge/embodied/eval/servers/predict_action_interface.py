# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared predict_action interface helpers for embodied eval model adapters."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PredictActionModel(Protocol):
    """Protocol for models that can be reused by the generic eval policy path."""

    def predict_action(
        self,
        images: Any,
        instructions: Any,
        state: Optional[Any] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return an action chunk for batched images and instructions."""


def validate_predict_action_model(model: Any) -> None:
    """Validate that a model exposes the eval-compatible predict_action method."""
    predict_action = getattr(model, "predict_action", None)
    if not callable(predict_action):
        raise TypeError("model must expose a callable predict_action(images, instructions, state, dataset_stats)")

    signature = inspect.signature(predict_action)
    parameters = signature.parameters
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    required = ("images", "instructions")
    optional = ("state", "dataset_stats")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise TypeError(f"model.predict_action is missing required parameters: {missing}")
    unsupported = [name for name in optional if name not in parameters and not has_kwargs]
    if unsupported:
        raise TypeError(f"model.predict_action cannot accept eval keyword parameters: {unsupported}")


def call_predict_action(
    model: PredictActionModel,
    images: Any,
    instructions: Any,
    state: Optional[Any],
    dataset_stats: Optional[Dict[str, Any]],
    action_dim: int,
) -> np.ndarray:
    """Call a compatible predict_action implementation and normalize its output shape."""
    validate_predict_action_model(model)
    result = model.predict_action(
        images=images,
        instructions=instructions,
        state=state,
        dataset_stats=dataset_stats,
    )
    actions = np.asarray(result, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    elif actions.ndim != 2:
        raise ValueError(f"model.predict_action returned unsupported action shape: {actions.shape}")

    if actions.shape[-1] < action_dim:
        raise ValueError(
            f"model.predict_action returned action dim {actions.shape[-1]}, expected at least {action_dim}"
        )
    return actions[:, :action_dim]
