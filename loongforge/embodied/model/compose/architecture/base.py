"""
Layer 2 - BaseArchitecture: Network structure base class

Abstract Factory pattern: defines how backbone + condition + action_loss are composed.
Three concrete factories correspond to three mainstream network paradigms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from model.compose.condition.base import BaseCondition
from model.compose.action.base import BaseActionLoss


class BaseArchitecture(ABC, nn.Module):
    """
    Network structure abstract base class.

    Responsibility: define how backbone combines with condition and action_loss.
    Subclasses implement different network paradigms:
      - VLMActionModel:        VLM backbone → condition → action_loss
      - WAM:                   Unified latent space model (action = latent frame)
      - WorldModelActionModel: World model encoder → condition → action_loss
    """

    def __init__(self, config, condition: BaseCondition, action_loss: BaseActionLoss):
        """
        Args:
            config: Full framework configuration (OmegaConf)
            condition: Modality alignment strategy instance (Layer 3)
            action_loss: Action loss strategy instance (Layer 4)
        """
        super().__init__()
        self.config = config
        self.condition = condition
        self.action_loss = action_loss

    @abstractmethod
    def encode(
        self,
        images: List[Any],
        instructions: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode observation inputs into feature representations.

        Args:
            images: Image list (PIL Image / Tensor, supports multi-view)
            instructions: Language instruction list

        Returns:
            Dict containing at least:
                'features': Tensor (B, L, H) — backbone main output
            Optional:
                'hidden_states': Tuple[Tensor] — per-layer hidden states (for layerwise alignment)
                'auxiliary_outputs': Dict — extra outputs (video loss, vlm loss, etc.)
        """
        ...

    @abstractmethod
    def forward(self, examples: List[Dict[str, Any]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Complete training forward pass.

        Args:
            examples: Batch list, each dict contains 'image', 'lang', 'action', optional 'state'

        Returns:
            Dict containing at least:
                'action_loss': scalar tensor
            Optional:
                'video_loss', 'vlm_loss', 'kl_loss', 'total_loss'
        """
        ...

    @abstractmethod
    def predict_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Inference: generate normalized actions.

        Returns:
            Dict containing:
                'normalized_actions': ndarray (B, T, action_dim)
        """
        ...

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        """Return feature extraction backbone (VLM / WorldModel / CNN)"""
        ...

    @property
    def action_head(self) -> nn.Module:
        """Return action prediction head (held by action_loss)"""
        return self.action_loss.action_head
