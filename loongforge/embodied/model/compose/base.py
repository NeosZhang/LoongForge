# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
ModelFramework - Top-level model wrapper after four-layer composition

Inherits nn.Module (does not depend on transformers.PreTrainedModel in standalone project),
compatible forward/predict_action interface
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from loongforge.embodied.model.compose.architecture.base import BaseArchitecture


class ModelFramework(nn.Module):
    """
    Top-level model after four-layer composition.

    Assembly relationship:
        ModelFramework
            └── architecture (BaseArchitecture)
                    ├── backbone (nn.Module)
                    ├── condition (BaseCondition)
                    └── action (BaseAction)
                            └── action_head (nn.Module)

        - forward(examples) -> Dict[str, Tensor]
        - predict_action(**kwargs) -> Dict[str, ndarray]
    """

    def __init__(self, architecture: BaseArchitecture, config):
        """
        Args:
            architecture: Assembled BaseArchitecture instance (with condition + action)
            config: Full framework configuration
        """
        super().__init__()
        self.architecture = architecture
        self.config = config

    def forward(self, examples: List[Dict[str, Any]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Training forward pass — delegates to architecture.

        Args:
            examples: Batch list, each dict contains 'image', 'lang', 'action', optional 'state'

        Returns:
            Dict containing 'action' (required) and other optional loss terms
        """
        return self.architecture.forward(examples, **kwargs)

    def predict_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Inference — delegates to architecture.

        Returns:
            Dict containing 'normalized_actions': ndarray (B, T, action_dim)
        """
        return self.architecture.predict_action(**kwargs)

    @property
    def backbone(self) -> nn.Module:
        """Get architecture's backbone."""
        return self.architecture.backbone

    @property
    def condition(self) -> nn.Module:
        """Get architecture's condition."""
        return self.architecture.condition

    @property
    def action(self) -> nn.Module:
        """Get architecture's action."""
        return self.architecture.action

    @property
    def action_head(self) -> nn.Module:
        """Get architecture's action_head."""
        return self.architecture.action_head
