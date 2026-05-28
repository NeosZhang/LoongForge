"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch

from .lerobot_types import EnvTransition, PolicyAction, TransitionKey
from .types import PipelineFeatureType, PolicyFeature


TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ProcessorStepRegistry:
    """Registry for processor step classes."""
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a processor step class."""
        def decorator(step_cls: type) -> type:
            cls._registry[name] = step_cls
            return step_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type | None:
        """Return the registered processor step class for the given name, or None."""
        return cls._registry.get(name)


@dataclass
class ProcessorStep(ABC, Generic[TInput, TOutput]):
    """Base class for processor steps."""

    def __call__(self, transition: Any) -> Any:
        """Invoke the processor step; must be implemented by subclasses."""
        raise NotImplementedError

    def transform_features(self, features: Any) -> Any:
        """Transform feature metadata; returns features unchanged by default."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return step configuration dict."""
        return {}


@dataclass
class PolicyProcessorPipeline(Generic[TInput, TOutput]):
    """Pipeline that processes policy data through a sequence of processor steps."""
    steps: list[ProcessorStep]
    name: str = "policy_processor"
    to_transition: Callable | None = None
    to_output: Callable | None = None

    def __call__(self, data: TInput) -> TOutput:
        """Process a transition through all registered steps sequentially."""
        result = data
        for step in self.steps:
            result = step(result)
        return result


@dataclass
class ObservationProcessorStep(ProcessorStep):
    """Base class for observation processor steps."""

    def observation(self, observation: dict) -> dict:
        """Process a single observation dict; returns it unchanged by default."""
        return observation

    def __call__(self, transition: Any) -> Any:
        """Apply observation processing to the transition's observation field."""
        obs = transition.get(TransitionKey.OBSERVATION, {})
        if obs is not None:
            processed = self.observation(obs)
            transition[TransitionKey.OBSERVATION] = processed
        return transition


@dataclass
class AddBatchDimensionProcessorStep(ProcessorStep):
    """Add batch dimension to observation tensors."""

    def __call__(self, transition: Any) -> Any:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is not None:
            obs = obs.copy()
            for key, value in obs.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.ndim < 5:
                    obs[key] = value.unsqueeze(0)
            new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        return features


@dataclass
class DeviceProcessorStep(ProcessorStep):
    """Move tensors to a specific device."""
    device: str = "cuda"

    def __call__(self, transition: Any) -> Any:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is not None:
            obs = obs.copy()
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    obs[key] = value.to(device=self.device)
            new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        return features


@dataclass
class NormalizerProcessorStep(ProcessorStep):
    """Normalize features using provided stats."""
    features: dict[str, PolicyFeature] | None = None
    norm_map: dict[str, Any] | None = None
    stats: dict[str, dict[str, torch.Tensor]] | None = None

    def __call__(self, transition: Any) -> Any:
        return transition

    def transform_features(self, features):
        return features


@dataclass
class RenameObservationsProcessorStep(ObservationProcessorStep):
    """Rename observation keys."""
    rename_map: dict[str, str] = field(default_factory=dict)

    def observation(self, observation: dict) -> dict:
        result = {}
        for key, value in observation.items():
            new_key = self.rename_map.get(key, key)
            result[new_key] = value
        return result


@dataclass
class TokenizerProcessorStep(ProcessorStep):
    """Tokenize language observations."""
    tokenizer_name: str = "facebook/bart-large"
    max_length: int = 64
    padding: str = "max_length"
    padding_side: str = "right"

    def __call__(self, transition: Any) -> Any:
        return transition

    def transform_features(self, features):
        return features


@dataclass
class UnnormalizerProcessorStep(ProcessorStep):
    """Unnormalize features."""
    features: dict[str, PolicyFeature] | None = None
    norm_map: dict[str, Any] | None = None
    stats: dict[str, dict[str, torch.Tensor]] | None = None

    def __call__(self, transition: Any) -> Any:
        return transition

    def transform_features(self, features):
        return features


def policy_action_to_transition(action: PolicyAction) -> EnvTransition:
    """Convert policy action to transition."""
    return {
        TransitionKey.ACTION: action,
    }


def transition_to_policy_action(transition: EnvTransition) -> PolicyAction:
    """Convert transition to policy action."""
    return transition.get(TransitionKey.ACTION)