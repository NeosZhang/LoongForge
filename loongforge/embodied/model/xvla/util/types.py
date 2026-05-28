"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

from dataclasses import dataclass
from enum import Enum


class FeatureType(str, Enum):
    """Enum of feature types for policy inputs and outputs."""
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"


class PipelineFeatureType(str, Enum):
    """Enum distinguishing action vs. observation pipeline features."""
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"


class NormalizationMode(str, Enum):
    """Supported normalization modes for policy features."""
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"


@dataclass
class PolicyFeature:
    """Describes a single policy feature: its type, tensor shape, key name, and dtype."""
    type: FeatureType
    shape: tuple[int, ...]
    key: str | None = None
    dtype: str = "float32"