"""
XVLA Data Processor and Pre/Post-processing Steps

Provides LeRobot-compatible processor pipelines for XVLA including:
- Image normalization and scaling
- Tokenization
- Domain ID injection
- Specialized processing for LIBERO environment
"""
from dataclasses import dataclass
from typing import Any

import torch

from loongforge.embodied.model.xvla.util.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from loongforge.embodied.model.xvla.util.lerobot_types import EnvTransition, PolicyAction, TransitionKey

from loongforge.embodied.model.xvla.configuration_xvla import XVLAConfig
from loongforge.embodied.model.xvla.util.constants import (
    IMAGENET_STATS,
    OBS_IMAGES,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_xvla_pre_post_processors(
    config: XVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA.
    """

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        XVLAAddDomainIdProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
@ProcessorStepRegistry.register(name="xvla_image_to_float")
class XVLAImageToFloatProcessorStep(ProcessorStep):
    """Convert image observations from [0, 255] to [0, 1] range.

    This processor step divides image observations by 255 to convert from uint8-like
    range [0, 255] to float range [0, 1]. This is typically used when loading images
    that are stored as uint8 values.

    Args:
        image_keys: List of observation keys that contain images to convert.
                   If None, will automatically detect keys starting with "observation.images."
        validate_range: If True, validates that input values are in [0, 255] range (default: True)

    Raises:
        ValueError: If validate_range is True and image values are not in [0, 255] range.
    """

    image_keys: list[str] | None = None
    validate_range: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert image observations from [0, 255] to [0, 1]."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # Make a copy of observations to avoid modifying the original
        obs = obs.copy()

        # Determine which keys to convert
        keys_to_convert = self.image_keys
        if keys_to_convert is None:
            # Auto-detect image keys
            keys_to_convert = [k for k in obs if k.startswith(OBS_IMAGES)]

        # Convert each image
        for key in keys_to_convert:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]

                min_val = tensor.min().item()
                max_val = tensor.max().item()

                if max_val <= 1.0:
                    obs[key] = tensor.float()  # ensure float dtype, but no division
                    continue
                # Validate that values are in [0, 255] range if requested
                if self.validate_range and (min_val < 0.0 or max_val > 255.0):
                    raise ValueError(
                        f"Image '{key}' has values outside [0, 255] range: "
                        f"min={min_val:.4f}, max={max_val:.4f}. "
                        f"Cannot convert to [0, 1] range."
                    )

                # Convert to float and divide by 255
                obs[key] = tensor.float() / 255.0

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """Image conversion doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "image_keys": self.image_keys,
            "validate_range": self.validate_range,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_imagenet_normalize")
class XVLAImageNetNormalizeProcessorStep(ProcessorStep):
    """Normalize image observations using ImageNet statistics.

    This processor step applies ImageNet normalization (mean and std) to image observations.
    It validates that input values are in the [0, 1] range before normalizing.

    The normalization formula is: (image - mean) / std

    Args:
        image_keys: List of observation keys that contain images to normalize.
                   If None, will automatically detect keys starting with "observation.images."

    Raises:
        ValueError: If image values are not in the [0, 1] range.
    """

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Normalize image observations using ImageNet statistics."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # Make a copy of observations to avoid modifying the original
        obs = obs.copy()

        # Determine which keys to normalize
        keys_to_normalize = self.image_keys
        if keys_to_normalize is None:
            # Auto-detect image keys
            keys_to_normalize = [k for k in obs if k.startswith(OBS_IMAGES)]

        # Normalize each image
        for key in keys_to_normalize:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]

                # Validate that values are in [0, 1] range
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                if min_val < 0.0 or max_val > 1.0:
                    raise ValueError(
                        f"Image '{key}' has values outside [0, 1] range: "
                        f"min={min_val:.4f}, max={max_val:.4f}. "
                        f"ImageNet normalization requires input values in [0, 1]."
                    )

                # Apply ImageNet normalization
                mean = torch.tensor(IMAGENET_STATS["mean"], device=tensor.device, dtype=tensor.dtype)
                std = torch.tensor(IMAGENET_STATS["std"], device=tensor.device, dtype=tensor.dtype)

                # Expand mean/std to match tensor dims (e.g., BCHW or BNCHW)
                while mean.dim() < tensor.dim():
                    mean = mean.unsqueeze(0)
                    std = std.unsqueeze(0)

                # Normalize: (image - mean) / std
                obs[key] = (tensor - mean) / std

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """ImageNet normalization doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "image_keys": self.image_keys,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_add_domain_id")
class XVLAAddDomainIdProcessorStep(ProcessorStep):
    """Add domain_id to complementary data.

    This processor step adds a domain_id tensor to the complementary data,
    which is used by XVLA to identify different robot embodiments or task domains.

    Args:
        domain_id: The domain ID to add (default: 3)
    """

    domain_id: int = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Add domain_id to complementary data."""
        new_transition = transition.copy()
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        comp = {} if comp is None else comp.copy()

        # Infer batch size from observation tensors
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        batch_size = 1
        if obs:
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break

        # Add domain_id tensor
        comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long)

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_transition

    def transform_features(self, features):
        """Domain ID addition doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "domain_id": self.domain_id,
        }
