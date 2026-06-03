# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Pi0FlowMatchingLoss - Pi0/Pi0.5 Flow Matching Loss Strategy

Differences from the generic FlowMatchingMSE:
  1. Action Expert is integrated inside the loss layer (because Pi0's forward pass
     requires jointly processing prefix and suffix tokens within the expert)
  2. Supports prefix-cache mode and joint-attention mode
  3. Uses KV cache for accelerated multi-step denoising during inference

Algorithm:
  Training:
    x_t = (1-t)*x_0 + t*eps    (t ~ Beta(1.5, 1.0))
    u_t = eps - x_0              (target velocity field)
    suffix = embed_suffix(state, x_t, t)
    v_t = expert([prefix; suffix])[-action_horizon:]
    L = MSE(v_t, u_t)

  Inference:
    x_T ~ N(0, I)
    for step in range(N):
        t = 1 - step/N
        v_t = expert([prefix; x_t], t)
        x_t = x_t + v_t * dt

Corresponding models: Pi0, Pi0.5
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from loongforge.embodied.model.compose.model_registry import ACTION_REGISTRY
from loongforge.embodied.model.compose.action.base import BaseAction


@ACTION_REGISTRY.register("Pi0FlowMatching")
class Pi0FlowMatchingLoss(BaseAction):
    """
    Pi0/Pi0.5 Flow Matching loss strategy.

    This is a Pi0-specific extension of FlowMatchingMSE:
      - action_head is Pi0ActionExpert (with suffix embedding logic)
      - compute_loss directly calls expert.compute_loss() (which internally handles prefix-suffix concatenation)
      - predict calls expert.sample_actions() (Euler ODE with KV cache)

    Note: The key difference between Pi0FlowMatchingLoss and FlowMatchingMSE is:
      FlowMatchingMSE assumes action_head is a standalone DiT that receives context + noisy_actions + t.
      Pi0FlowMatchingLoss's action_head is the complete Pi0ActionExpert,
      which manages suffix construction and prefix-suffix joint forward on its own.
    """

    def __init__(self, config, action_head: nn.Module):
        super().__init__(config, action_head)
        self.action_dim = config.get("action_dim", 7)
        self.action_horizon = config.get("action_horizon", 10)
        self.noise_beta_alpha = config.get("noise_beta_alpha", 1.5)
        self.noise_beta_beta = config.get("noise_beta_beta", 1.0)
        self.num_inference_steps = config.get("num_inference_steps", 10)
        self.repeated_diffusion_steps = config.get("repeated_diffusion_steps", 1)

    def compute_loss(
        self,
        action_context: torch.Tensor,
        target_actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Pi0 Flow Matching training loss.

        Args:
            action_context: Dict from Pi0JointAttention.align():
                'prefix_embs': (B, L, W)
                'prefix_pad_masks': (B, L)
                'prefix_att_masks': (B, L)
                'vlm_language_model': Optional[nn.Module]
                'mode': str
            target_actions: (B, T, action_dim) ground truth actions
            state: (B, state_dim) optional

        Returns:
            Dict:
                'action_loss': scalar loss
                'flow_matching_loss': same (for logging)
        """
        # Unpack action_context
        if isinstance(action_context, dict):
            prefix_embs = action_context["prefix_embs"]
            prefix_pad_masks = action_context["prefix_pad_masks"]
            # Joint Attention mode: pass VLM language model to expert
            vlm_language_model = action_context.get("vlm_language_model", None)
        else:
            # Compatible with simple Tensor input (B, L, H)
            prefix_embs = action_context
            prefix_pad_masks = torch.ones(
                prefix_embs.shape[:2], dtype=torch.bool, device=prefix_embs.device
            )
            vlm_language_model = None

        B, T, A = target_actions.shape
        device = target_actions.device

        # Repeated diffusion steps data augmentation
        repeat = self.repeated_diffusion_steps
        if repeat > 1:
            target_actions = target_actions.repeat(repeat, 1, 1)
            prefix_embs = prefix_embs.repeat(repeat, 1, 1)
            prefix_pad_masks = prefix_pad_masks.repeat(repeat, 1)
            if state is not None:
                state = state.repeat(repeat, 1) if state.dim() == 2 else state.repeat(repeat, 1, 1)
            B = B * repeat

        # Call action expert's compute_loss
        # Automatic mode selection:
        #   - vlm_language_model is not None -> Joint Attention (_shared_forward)
        #   - vlm_language_model is None -> Prefix Cache (expert_forward)
        loss = self.action_head.compute_loss(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            target_actions=target_actions.to(device),
            state=state,
            vlm_language_model=vlm_language_model,
        )

        loss_mean = loss.mean()
        return {
            "action_loss": loss_mean,
            "flow_matching_loss": loss_mean.item(),
        }

    def predict(
        self,
        action_context: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Pi0 inference: Euler ODE multi-step denoising.

        Returns:
            (B, action_horizon, action_dim) predicted actions
        """
        # Unpack action_context
        if isinstance(action_context, dict):
            prefix_embs = action_context["prefix_embs"]
            prefix_pad_masks = action_context["prefix_pad_masks"]
            vlm_language_model = action_context.get("vlm_language_model", None)
        else:
            prefix_embs = action_context
            prefix_pad_masks = torch.ones(
                prefix_embs.shape[:2], dtype=torch.bool, device=prefix_embs.device
            )
            vlm_language_model = None

        num_steps = kwargs.get("num_steps", self.num_inference_steps)

        # Call action expert's sample_actions
        pred_actions = self.action_head.sample_actions(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            state=state,
            num_steps=num_steps,
            vlm_language_model=vlm_language_model,
        )

        return pred_actions
