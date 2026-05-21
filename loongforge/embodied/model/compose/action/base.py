"""
Layer 4 - BaseActionLoss: Action Loss Strategy Base Class

Strategy pattern: defines the training loss computation and inference action generation
interface for the action head. Each strategy encapsulates specific noise scheduling,
loss weighting, and denoising sampling logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn


class BaseActionLoss(ABC, nn.Module):
    """
    Action loss strategy abstract base class.

    Responsibilities:
      1. Training: given action_context + target_actions, compute scalar loss
      2. Inference: given action_context, generate predicted actions

    Note: ActionLoss holds action_head (nn.Module) because loss computation
    is typically deeply coupled with the action head's forward pass (e.g.,
    flow matching requires injecting noise inside forward).
    """

    def __init__(self, config, action_head: nn.Module):
        """
        Args:
            config: action_model sub-configuration
            action_head: the actual action prediction network (DiT, MLP, Transformer, etc.)
        """
        super().__init__()
        self.config = config
        self.action_head = action_head

    @abstractmethod
    def compute_loss(
        self,
        action_context: torch.Tensor,
        target_actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            action_context: output from condition.inject()
                          (Tensor / List[Tensor] / Dict, depends on alignment strategy)
            target_actions: ground truth actions (B, T, action_dim)
            state: optional robot state (B, state_dim) or (B, T, state_dim)

        Returns:
            Dict containing at least:
                'action_loss': scalar tensor -- main loss (required)
            Optional:
                'kl_loss': KL divergence for CVAE
                'velocity_loss': Flow matching velocity field loss
                'requires_stdp': bool -- whether STDP update is needed in on_step_end
        """
        ...

    @abstractmethod
    def predict(
        self,
        action_context: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Inference: generate predicted actions from action_context.

        Args:
            action_context: output from condition.inject()
            state: optional robot state

        Returns:
            Tensor (B, chunk_len, action_dim) -- predicted normalized actions
        """
        ...

    def extra_repr(self) -> str:
        """add extra info"""
        return f"action_head={self.action_head.__class__.__name__}"
