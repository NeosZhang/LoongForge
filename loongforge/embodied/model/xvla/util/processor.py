"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

from .lerobot_types import EnvTransition, TransitionKey
from .processor_pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
    PolicyProcessorPipeline,
    ObservationProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)

# Type alias for policy action
PolicyAction = ...  # Will be filled from lerobot_types

__all__ = [
    "EnvTransition",
    "TransitionKey",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "PolicyProcessorPipeline",
    "ObservationProcessorStep",
    "AddBatchDimensionProcessorStep",
    "DeviceProcessorStep",
    "NormalizerProcessorStep",
    "RenameObservationsProcessorStep",
    "TokenizerProcessorStep",
    "UnnormalizerProcessorStep",
    "policy_action_to_transition",
    "transition_to_policy_action",
]