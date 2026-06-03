# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
WorldModelActionModel - World Model + Action Model (Decoupled Architecture)

Architecture: World model encoder extracts visual spatiotemporal features -> condition -> action_loss
Corresponding models: WorldModelVLA (Cosmos-Predict2 / V-JEPA2 / WAN)

Difference from VLMActionModel: backbone is a video generation/prediction model, not a VLM
Difference from WAM: action head is independent (decoupled), does not share latent space
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from loongforge.embodied.model.compose.model_registry import ARCHITECTURE_REGISTRY
from loongforge.embodied.model.compose.architecture.base import BaseArchitecture
from loongforge.embodied.model.compose.condition.base import BaseCondition
from loongforge.embodied.model.compose.action.base import BaseAction


@ARCHITECTURE_REGISTRY.register("WorldModelActionModel")
class WorldModelActionModel(BaseArchitecture):
    """
    WorldModel + ActionModel decoupled architecture.

    Data flow:
      images + text (T5 precomputed)
            |
            v
      ┌──────────────┐
      │ World Model  │  (Cosmos-Predict2 / V-JEPA2 / WAN)
      │   Encoder    │  Extracts spatiotemporal visual features + optional video prediction loss
      └──────┬───────┘
             │  visual_features (B, N, D) + optional video_loss
             v
      ┌──────────────┐
      │  Condition   │  (CrossAttentionFusion: visual attend to text)
      │   (Layer3)   │
      └──────┬───────┘
             │  fused_context
             v
      ┌──────────────┐
      │    Action    │  (FlowMatchingMSE: DiT-based action generation)
      │   (Layer4)   │
      └──────┬───────┘
             │  actions
             v
    """

    def __init__(self, config, condition: BaseCondition, action: BaseAction):
        super().__init__(config, condition, action)

        wm_cfg = config.framework.get("world_model", {})
        self.feature_layer_id = wm_cfg.get("feature_layer_id", 18)
        self.use_video_loss = wm_cfg.get("use_video_loss", True)
        self.video_loss_weight = wm_cfg.get("video_loss_weight", 0.1)

        # World Model Encoder
        self._backbone: Optional[nn.Module] = None
        # Text Encoder (T5)
        self._text_encoder: Optional[nn.Module] = None

    @property
    def backbone(self) -> nn.Module:
        """Get World Model encoder."""
        if self._backbone is None:
            raise RuntimeError("World model encoder not initialized.")
        return self._backbone

    def set_backbone(self, backbone: nn.Module):
        """Set World Model encoder."""
        self._backbone = backbone

    def set_text_encoder(self, text_encoder: nn.Module):
        """Set T5 text encoder."""
        self._text_encoder = text_encoder

    def encode(
        self,
        images: List[Any],
        instructions: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        World Model encoding: extract intermediate layer spatiotemporal features.

        Returns:
            'features': (B, N, D) -- visual features from specified layer
            'hidden_states': Tuple[Tensor] -- optional layer-wise outputs
            'text_embeddings': (B, L, D_text) -- T5 text embeddings
            'video_loss': scalar -- optional next-frame prediction loss
        """
        text_emb = kwargs.get("text_embeddings")

        # World model forward (extract intermediate layer features)
        if hasattr(self.backbone, "forward_all_layers"):
            outputs = self.backbone.forward_all_layers(images)
            features = outputs["layer_features"][self.feature_layer_id]
            hidden_states = outputs.get("all_layer_features")
        elif hasattr(self.backbone, "forward_with_video_loss"):
            features, video_loss = self.backbone.forward_with_video_loss(
                images, text_emb
            )
            return {
                "features": features,
                "text_embeddings": text_emb,
                "video_loss": video_loss,
            }
        else:
            features = self.backbone(images)

        result = {
            "features": features,
            "text_embeddings": text_emb,
        }
        if hidden_states is not None:
            result["hidden_states"] = hidden_states
        return result

    def forward(self, examples: List[Dict[str, Any]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        WorldModel + ActionModel training forward.
        """
        images = [ex["image"] for ex in examples]
        actions = torch.stack([
            torch.as_tensor(ex["action"], dtype=torch.float32)
            for ex in examples
        ])
        text_emb = torch.stack([
            torch.as_tensor(ex.get("text_embeddings", ex.get("t5_text_embeddings")),
                           dtype=torch.float32)
            for ex in examples
        ]) if "text_embeddings" in examples[0] or "t5_text_embeddings" in examples[0] else None

        state = None
        if "state" in examples[0] and examples[0]["state"] is not None:
            state = torch.stack([
                torch.as_tensor(ex["state"], dtype=torch.float32)
                for ex in examples
            ])

        # 1. Encode (World Model)
        backbone_output = self.encode(images, None, text_embeddings=text_emb)

        # 2. condition (CrossAttentionFusion)
        aligned = self.condition.inject(backbone_output)

        # 3. Action Loss (FlowMatchingMSE)
        action_context = aligned["action_context"]
        loss_dict = self.action.compute_loss(
            action_context=action_context,
            target_actions=actions.to(action_context.device),
            state=state,
        )

        # 4. Add video prediction auxiliary loss
        video_loss = backbone_output.get("video_loss")
        if video_loss is not None and self.use_video_loss:
            action_loss = loss_dict["action_loss"]
            loss_dict["video_loss"] = video_loss
            loss_dict["total_loss"] = action_loss + self.video_loss_weight * video_loss

        return loss_dict

    def predict_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """Inference: encode -> align -> predict."""
        images = kwargs.get("images")
        text_emb = kwargs.get("text_embeddings")
        state = kwargs.get("state")

        backbone_output = self.encode(images, None, text_embeddings=text_emb)
        aligned = self.condition.inject(backbone_output)
        pred = self.action.predict(
            action_context=aligned["action_context"],
            state=state,
        )

        return {"normalized_actions": pred.cpu().numpy()}
