"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

import abc
import builtins
import json
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from .types import FeatureType, PolicyFeature
from .device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from .hub import HubMixin
from .constants import ACTION, OBS_STATE

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base configuration for optimizers."""

    lr: float
    weight_decay: float
    grad_clip_norm: float

    @abc.abstractmethod
    def build(self, params) -> Any:
        """Build and return the optimizer from the given parameters."""
        raise NotImplementedError


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base configuration for learning rate schedulers."""

    num_warmup_steps: int | None

    @abc.abstractmethod
    def build(self, optimizer, num_training_steps: int) -> Any:
        """Build and return the LR scheduler."""
        raise NotImplementedError


@dataclass
class PreTrainedConfig(  # type: ignore[misc,name-defined] #TODO: draccus issue
    draccus.ChoiceRegistry, HubMixin, abc.ABC
):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
    """

    n_obs_steps: int = 1

    # `input_features` can be set to None/null in order to infer those values from the dataset.
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    # Whether the policy employed PEFT for training.
    use_peft: bool = False

    push_to_hub: bool = True
    repo_id: str | None = None

    # Upload on private repository on the Hugging Face hub.
    private: bool | None = None
    # Add tags to your policy on the hub.
    tags: list[str] | None = None
    # Add tags to your policy on the hub.
    license: str | None = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    pretrained_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate and auto-select device; deactivate AMP if unsupported."""
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def observation_delta_indices(self) -> list | None:
        """Return observation delta indices; must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def action_delta_indices(self) -> list | None:
        """Return action delta indices; must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def reward_delta_indices(self) -> list | None:
        """Return reward delta indices; must be implemented by subclasses."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        """Return the optimizer preset config for this policy."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Return the LR scheduler preset config for this policy."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        """Validate feature config; raise if invalid."""
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """Return the robot proprioceptive state feature, if present."""
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        """Return the environment state feature, if present."""
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Return all VISUAL input features."""
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        """Return the ACTION output feature, if present."""
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save config as JSON into save_directory."""
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs: Any,
    ) -> T:
        """Load a PreTrainedConfig from a local directory or HuggingFace Hub repo."""
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            config = json.load(f)

        return cls(**config)