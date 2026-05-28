"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

from .types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from .lerobot_types import EnvTransition, PolicyAction, TransitionKey
from .constants import (
    ACTION,
    IMAGENET_STATS,
    OBS_IMAGES,
    OBS_LANGUAGE_TOKENS,
    OBS_PREFIX,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from .import_utils import _transformers_available
from .optimizers import XVLAAdamWConfig
from .schedulers import CosineDecayWithWarmupSchedulerConfig
from .processor import (
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

# Re-export from lerobot_types
from .lerobot_types import PolicyAction
from .policies import OptimizerConfig, LRSchedulerConfig, PreTrainedConfig
from .device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available

__all__ = [
    # Types
    "FeatureType",
    "NormalizationMode",
    "PipelineFeatureType",
    "PolicyFeature",
    "EnvTransition",
    "PolicyAction",
    "TransitionKey",
    # Constants
    "ACTION",
    "IMAGENET_STATS",
    "OBS_IMAGES",
    "OBS_LANGUAGE_TOKENS",
    "OBS_PREFIX",
    "OBS_STATE",
    "POLICY_POSTPROCESSOR_DEFAULT_NAME",
    "POLICY_PREPROCESSOR_DEFAULT_NAME",
    # Import utils
    "_transformers_available",
    "auto_select_torch_device",
    "is_amp_available",
    "is_torch_device_available",
    # Optimizer & Scheduler
    "OptimizerConfig",
    "LRSchedulerConfig",
    "XVLAAdamWConfig",
    "CosineDecayWithWarmupSchedulerConfig",
    # Config
    "PreTrainedConfig",
    # Processor
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