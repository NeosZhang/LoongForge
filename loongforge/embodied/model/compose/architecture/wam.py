# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
WAM - World-Action Model (Unified Latent Space Modeling)

Architecture: A unified Video DiT model where actions are encoded as latent frames,
             jointly denoised with observation frames in the same latent space.
             No separate action head.

Corresponding model: CosmosPolicy
Key feature: action = latent frame 4, the entire model is a video diffusion model
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from loongforge.embodied.model.compose.registry import ARCHITECTURE_REGISTRY
from loongforge.embodied.model.compose.architecture.base import BaseArchitecture
from loongforge.embodied.model.compose.condition.base import BaseCondition
from loongforge.embodied.model.compose.action.base import BaseAction


@ARCHITECTURE_REGISTRY.register("WAM")
class WAM(BaseArchitecture):
    """
    World-Action Model (WAM) unified latent space architecture.

    Fundamental difference from VLMActionModel:
      - VLMActionModel: VLM -> features -> action head -> actions
      - WAM: [observations + actions] -> latent -> unified DiT -> denoised latent -> extract actions

    Data flow:
      video frames + actions + state
            |
            v
      ┌──────────┐
      │  VAE     │  (WAN 2.1 VAE, frozen)
      │ Encoder  │
      └────┬─────┘
           │  latent (B, C, T, H, W), action occupies specific frame positions
           v
      ┌──────────┐
      │Condition │  (LatentFrameInjection: builds condition mask)
      │ (Layer3) │
      └────┬─────┘
           │  latent + mask + text_emb
           v
      ┌──────────┐
      │  Action  │  (EDMRectifiedFlow: unified denoising)
      │ (Layer4) │  DiT denoises the entire latent, action frames are the target
      └────┬─────┘
           │  denoised latent -> extract action frame -> reshape
           v
        actions
    """

    def __init__(self, config, condition: BaseCondition, action: BaseAction):
        super().__init__(config, condition, action)

        cosmos_cfg = config.framework.get("cosmos_policy", {})
        self.action_frame_idx = cosmos_cfg.get("action_frame_idx", 4)
        self.action_dim = cosmos_cfg.get("action_dim", 7)
        self.chunk_size = cosmos_cfg.get("chunk_size", 16)
        self.sigma_data = cosmos_cfg.get("sigma_data", 1.0)

        # VAE Encoder (frozen)
        self._vae_encoder: Optional[nn.Module] = None
        # Text Encoder (T5, precomputed)
        self._text_encoder: Optional[nn.Module] = None
        # Video DiT backbone
        self._backbone: Optional[nn.Module] = None

    @property
    def backbone(self) -> nn.Module:
        """backbone"""
        if self._backbone is None:
            raise RuntimeError("DiT backbone not initialized. Call set_backbone().")
        return self._backbone

    def set_backbone(self, backbone: nn.Module):
        """Set Video DiT backbone."""
        self._backbone = backbone

    def set_vae_encoder(self, vae: nn.Module):
        """Set VAE encoder (frozen)."""
        self._vae_encoder = vae
        for param in vae.parameters():
            param.requires_grad = False

    def encode(
        self,
        images: List[Any],
        instructions: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        VAE encode video -> latent volume + T5 text embeddings.

        Returns:
            'latent_volume': (B, C, T, H, W)
            'text_embeddings': (B, L, D_text)
        """
        # VAE encode
        video_tensor = kwargs.get("video")  # (B, 3, num_frames, H, W)
        if self._vae_encoder is not None and video_tensor is not None:
            with torch.no_grad():
                latent = self._vae_encoder(video_tensor) * self.sigma_data
        else:
            latent = kwargs.get("latent_volume")  # Pre-encoded

        # Text embeddings (precomputed T5)
        text_emb = kwargs.get("text_embeddings")

        return {
            "latent_volume": latent,
            "text_embeddings": text_emb,
            "features": latent,  # BaseArchitecture interface compatibility
        }

    def forward(self, examples: List[Dict[str, Any]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        WAM training forward:
          1. VAE encode -> latent
          2. Inject action/state into specific latent frames
          3. Add noise -> DiT denoise -> compute action frame loss
        """
        # Extract data
        video = torch.stack([torch.as_tensor(ex["video"], dtype=torch.float32) for ex in examples])
        actions = torch.stack([torch.as_tensor(ex["action"], dtype=torch.float32) for ex in examples])
        text_emb = torch.stack([torch.as_tensor(ex["t5_text_embeddings"], dtype=torch.float32) for ex in examples])

        # 1. Encode
        backbone_output = self.encode(
            images=None, instructions=None,
            video=video, text_embeddings=text_emb,
        )

        # 2. Condition (build condition mask + inject actions into latent)
        aligned = self.condition.inject(backbone_output)

        # 3. Action loss (EDM denoising loss, only for action frames)
        action_context = aligned["action_context"]
        loss_dict = self.action.compute_loss(
            action_context=action_context,
            target_actions=actions,
        )

        return loss_dict

    def predict_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        WAM inference: denoise latent -> extract action frame -> reshape to actions.
        """
        video = kwargs.get("video")
        text_emb = kwargs.get("text_embeddings")

        # 1. Encode
        backbone_output = self.encode(
            images=None, instructions=None,
            video=video, text_embeddings=text_emb,
        )

        # 2. Condition
        aligned = self.condition.inject(backbone_output)

        # 3. Predict (denoise -> extract action frame)
        action_context = aligned["action_context"]
        pred_latent = self.action.predict(
            action_context=action_context,
            action_shape=(video.shape[0], self.chunk_size, self.action_dim),
        )

        return {"normalized_actions": pred_latent.cpu().numpy()}
