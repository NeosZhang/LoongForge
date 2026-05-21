"""
LayeredFrameworkBuilder - Builder pattern assembler

Builds the four-layer framework step by step from OmegaConf config:
    condition → action_loss → architecture → LayeredFramework
"""

from typing import Optional
import importlib
import pkgutil
import torch.nn as nn

from model.compose.registry import (
    ARCHITECTURE_REGISTRY,
    CONDITION_REGISTRY,
    LOSS_REGISTRY,
    TRAINER_REGISTRY,
)
from model.compose.base import LayeredFramework
from model.compose.condition.base import BaseCondition
from model.compose.action.base import BaseActionLoss
from model.compose.architecture.base import BaseArchitecture


def _auto_import_submodules(package_path: str, package_name: str):
    """Auto-import submodules to trigger @register decorators."""
    for _, module_name, _ in pkgutil.iter_modules([package_path]):
        if module_name != "base":
            importlib.import_module(f"{package_name}.{module_name}")


class LayeredFrameworkBuilder:
    """
    Builder pattern: build the complete four-layer training framework step by step from config.

    Usage:
        builder = LayeredFrameworkBuilder(cfg)
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
        self._action_loss: Optional[BaseActionLoss] = None
        self._architecture: Optional[BaseArchitecture] = None

    def build_condition(self) -> "LayeredFrameworkBuilder":
        """Step 1: Build condition injection strategy from config."""
        condition_name = self.cfg.framework.compose.condition
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

    def build_action(self) -> "LayeredFrameworkBuilder":
        """Step 2: Build action loss strategy + action head from config."""
        loss_name = self.cfg.framework.compose.action_loss
        # Auto-import all action_loss implementations
        import model.compose.action as loss_pkg
        import os
        _auto_import_submodules(
            os.path.dirname(loss_pkg.__file__),
            "model.compose.action_loss",
        )
        LossCls = LOSS_REGISTRY[loss_name]
        # Build raw action head module
        action_head = self._build_action_head_module()
        self._action_loss = LossCls(
            config=self.cfg.framework.action_model,
            action_head=action_head,
        )
        return self

    def build_architecture(self) -> "LayeredFrameworkBuilder":
        """Step 3: Assemble backbone + condition + action_loss from config."""
        arch_name = self.cfg.framework.compose.architecture
        # Auto-import all architecture implementations
        import model.compose.architecture as arch_pkg
        import os
        _auto_import_submodules(
            os.path.dirname(arch_pkg.__file__),
            "model.compose.architecture",
        )
        ArchCls = ARCHITECTURE_REGISTRY[arch_name]
        self._architecture = ArchCls(
            config=self.cfg,
            condition=self._condition,
            action_loss=self._action_loss,
        )
        return self

    def validate(self) -> "LayeredFrameworkBuilder":
        """Step 4: Validate compatibility between layers (dimension matching, etc.)."""
        assert self._condition is not None, "Must call build_condition() first"
        assert self._action_loss is not None, "Must call build_action_loss() first"
        assert self._architecture is not None, "Must call build_architecture() first"

        # Validate condition output compatibility with action_loss expectations
        spec = self._condition.get_action_head_input_spec()
        # Basic type check (specific strategies can extend with more refined validation)
        if spec.get("type") is None:
            raise ValueError("Condition must declare output type in get_action_head_input_spec()")

        return self

    def finalize(self) -> LayeredFramework:
        """Step 5: Produce the final assembled model."""
        return LayeredFramework(
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


def build_framework(cfg) -> LayeredFramework:
    """
    Convenience function to build the layered framework in one call.

    Args:
        cfg: OmegaConf configuration containing cfg.framework.compose

    Returns:
        LayeredFramework instance
    """
    builder = LayeredFrameworkBuilder(cfg)
    return (
        builder
        .build_condition()
        .build_action()
        .build_architecture()
        .validate()
        .finalize()
    )
