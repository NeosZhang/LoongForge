"""
ModelFrameworkBuilder - Builder pattern assembler

Builds the four-layer framework step by step from OmegaConf config:
    condition → action → architecture → ModelFramework
"""

from typing import Optional
import importlib
import pkgutil
import torch.nn as nn

from model.compose.registry import (
    ARCHITECTURE_REGISTRY,
    CONDITION_REGISTRY,
    ACTION_REGISTRY,
    TRAINER_REGISTRY,
)
from model.compose.base import ModelFramework
from model.compose.condition.base import BaseCondition
from model.compose.action.base import BaseAction
from model.compose.architecture.base import BaseArchitecture


def _auto_import_submodules(package_path: str, package_name: str):
    """Auto-import submodules to trigger @register decorators."""
    for _, module_name, _ in pkgutil.iter_modules([package_path]):
        if module_name != "base":
            importlib.import_module(f"{package_name}.{module_name}")


class ModelFrameworkBuilder:
    """
    Builder pattern: build the complete four-layer training framework step by step from config.

    Usage:
        builder = ModelFrameworkBuilder(cfg)
        model = (
            builder
            .build_condition()
            .build_action()
            .build_architecture()
            .validate()
            .finalize()
        )
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: OmegaConf full configuration, must contain cfg.framework.compose sub-node
        """
        self.cfg = cfg
        self._condition: Optional[BaseCondition] = None
        self._action: Optional[BaseAction] = None
        self._architecture: Optional[BaseArchitecture] = None

    def build_condition(self) -> "ModelFrameworkBuilder":
        """Step 1: Build condition injection strategy from config."""
        condition_name = self.cfg.framework.compose.condition
        if (condition_name is None or
                (isinstance(condition_name, str) and
                 condition_name.lower() in ("none", ""))):
            self._condition = None
            return self
        # Auto-import all condition implementations
        import model.compose.condition as condition_pkg
        import os
        _auto_import_submodules(
            os.path.dirname(condition_pkg.__file__),
            "model.compose.condition",
        )
        condition_cls = CONDITION_REGISTRY[condition_name]
        self._condition = condition_cls(config=self.cfg)
        return self

    def build_action(self) -> "ModelFrameworkBuilder":
        """Step 2: Build action loss strategy + action head from config."""
        action_name = self.cfg.framework.compose.get("action", None)
        if (action_name is None or
                (isinstance(action_name, str) and
                 action_name.lower() in ("none", ""))):
            self._action = None
            return self
        # Auto-import all action implementations
        import model.compose.action as action_pkg
        import os
        _auto_import_submodules(
            os.path.dirname(action_pkg.__file__),
            "model.compose.action",
        )
        action_cls = ACTION_REGISTRY[action_name]
        # Build raw action head module
        action_head = self._build_action_head_module()
        self._action = action_cls(
            config=self.cfg.framework.action_model,
            action_head=action_head,
        )
        return self

    def build_architecture(self) -> "ModelFrameworkBuilder":
        """Step 3: Assemble backbone + condition + action from config."""
        arch_name = self.cfg.framework.compose.architecture
        # Auto-import all architecture implementations
        import model.compose.architecture as arch_pkg
        import os
        _auto_import_submodules(
            os.path.dirname(arch_pkg.__file__),
            "model.compose.architecture",
        )
        arch_cls = ARCHITECTURE_REGISTRY[arch_name]
        self._architecture = arch_cls(
            config=self.cfg,
            condition=self._condition,
            action=self._action,
        )
        return self

    def validate(self) -> "ModelFrameworkBuilder":
        """Step 4: Validate compatibility between layers (dimension matching, etc.)."""
        assert self._architecture is not None, "Must call build_architecture() first"

        if self._condition is None or self._action is None:
            return self
        # Validate condition output compatibility with action_loss expectations
        spec = self._condition.get_action_head_input_spec()
        # Basic type check (specific strategies can extend with more refined validation)
        if spec.get("type") is None:
            raise ValueError("Condition must declare output type in get_action_head_input_spec()")

        return self

    def finalize(self) -> ModelFramework:
        """Step 5: Produce the final assembled model."""
        return ModelFramework(
            architecture=self._architecture,
            config=self.cfg,
        )

    def _build_action_head_module(self) -> nn.Module:
        """
        Build raw action head nn.Module.

        Determines the specific network based on config.framework.action_model.action_model_type:
          - "Pi0Expert" → Pi0ActionExpert
        """
        action_cfg = self.cfg.framework.action_model
        model_type = action_cfg.get("action_model_type", "MLPResNet")

        if model_type == "Pi0Expert":
            from model.modules.pi0_action_expert import Pi0ActionExpert
            return Pi0ActionExpert(
                action_dim=action_cfg.get("action_dim", 7),
                state_dim=action_cfg.get("state_dim", 7),
                action_horizon=action_cfg.get("action_horizon", 10),
                expert_width=action_cfg.get("action_expert_width", 1024),
                expert_depth=action_cfg.get("action_expert_depth", 18),
                expert_mlp_dim=action_cfg.get("action_expert_mlp_dim", 4096),
                expert_num_heads=action_cfg.get("action_expert_num_heads", 8),
                expert_head_dim=action_cfg.get("action_expert_head_dim", 128),
                pi05=self.cfg.framework.get("pi05", True),
                num_inference_steps=action_cfg.get("num_inference_steps", 10),
                noise_beta_alpha=action_cfg.get("noise_beta_alpha", 1.5),
                noise_beta_beta=action_cfg.get("noise_beta_beta", 1.0),
            )

        # Other types: dispatch skeleton, specific implementations need to import corresponding modules
        # Reuse existing factory
        raise NotImplementedError(
            f"action_head construction for '{model_type}' should be implemented "
            f"by registering a head builder or integrating with AlphaBrain's "
            f"get_action_model() factory."
        )


def build_framework(cfg) -> ModelFramework:
    """build_framework
    Convenience function to build the model framework in one call.

    Args:
        cfg: OmegaConf configuration containing cfg.framework.compose

    Returns:
        ModelFramework instance
    """
    builder = ModelFrameworkBuilder(cfg)
    return (
        builder
        .build_condition()
        .build_action()
        .build_architecture()
        .validate()
        .finalize()
    )
