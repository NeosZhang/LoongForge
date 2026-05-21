"""
VLMActionModel - VLM + Action Model Architecture

Architecture: VLM backbone (frozen/fine-tuned) -> condition -> action_loss
Corresponding models: GR00T, Pi0/Pi0.5

This is the most general architecture paradigm: a large vision-language model extracts
multimodal features, which are transformed through a modality alignment layer and then
fed into an action head to generate actions.
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from model.compose.registry import ARCHITECTURE_REGISTRY
from model.compose.architecture.base import BaseArchitecture
from model.compose.condition.base import BaseCondition
from model.compose.action.base import BaseActionLoss


@ARCHITECTURE_REGISTRY.register("VLMActionModel")
class VLMActionModel(BaseArchitecture):
    """
    VLM + ActionModel architecture.

    Data flow:
      images + instructions
            |
            v
      ┌──────────┐
      │   VLM    │  (Qwen2.5-VL / Llama / PaliGemma / ResNet)
      │ backbone │
      └────┬─────┘
           │  features / hidden_states
           v
      ┌──────────┐
      │Condition │  (GlobalProj / Layerwise / KVCache / QFormer / Direct)
      │ (Layer3) │
      └────┬─────┘
           │  action_context
           v
      ┌──────────┐
      │ActionLoss│  (L1 / FlowMatching / CVAE / HybridSTDP)
      │ (Layer4) │
      └────┬─────┘
           │  action_loss / predicted_actions
           v
    """

    def __init__(self, config, condition: BaseCondition, action_loss: BaseActionLoss):
        super().__init__(config, condition, action_loss)

        # VLM backbone (created by get_vlm_model(config) during integration)
        # Uses placeholder in standalone project
        self._backbone: Optional[nn.Module] = None

        # Configuration
        vlm_cfg = config.framework.get("qwenvl", {})
        self.vlm_hidden_dim = vlm_cfg.get("vl_hidden_dim", 2048)
        self.output_hidden_states = True  # Needed for layer-wise alignment

    @property
    def backbone(self) -> nn.Module:
        """backbone"""
        if self._backbone is None:
            raise RuntimeError(
                "VLM backbone not initialized. Call set_backbone() "
            )
        return self._backbone

    def set_backbone(self, backbone: nn.Module):
        """Set VLM backbone (for external injection or integration)."""
        self._backbone = backbone

    def encode(
        self,
        images: List[Any],
        instructions: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        VLM encoding: images + instructions -> features + hidden_states.

        Returns:
            'features': Tensor (B, L, H) -- last layer output
            'hidden_states': Tuple[Tensor] -- all layer outputs (for layerwise)
            'attention_mask': Tensor (B, L) -- padding mask
        """
        # Interface contract defined here; actual VLM call is implemented during integration
        # backbone must implement forward(**inputs, output_hidden_states=True)
        outputs = self.backbone(
            images=images,
            instructions=instructions,
            output_hidden_states=self.output_hidden_states,
            **kwargs,
        )

        result = {"features": outputs["last_hidden_state"]}
        if "hidden_states" in outputs:
            result["hidden_states"] = outputs["hidden_states"]
        if "attention_mask" in outputs:
            result["attention_mask"] = outputs["attention_mask"]
        return result

    def forward(self, examples: List[Dict[str, Any]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Full training forward: encode -> condition -> compute_loss.
        """
        # Extract batch data
        images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = torch.stack([
            torch.as_tensor(ex["action"], dtype=torch.float32)
            for ex in examples
        ])  # (B, T, action_dim)

        state = None
        if "state" in examples[0] and examples[0]["state"] is not None:
            state = torch.stack([
                torch.as_tensor(ex["state"], dtype=torch.float32)
                for ex in examples
            ])

        # 1. Encode
        backbone_output = self.encode(images, instructions)

        # 2. Align
        align_kwargs = {}
        if state is not None and hasattr(self.condition, "film_modulator"):
            align_kwargs["state_history"] = state
        aligned = self.condition.inject(backbone_output, **align_kwargs)

        # 3. Compute loss
        action_context = aligned["action_context"]
        loss_dict = self.action_loss.compute_loss(
            action_context=action_context,
            target_actions=actions.to(action_context.device
                                      if isinstance(action_context, torch.Tensor)
                                      else actions.device),
            state=state,
        )

        return loss_dict

    def predict_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Inference: encode -> align -> predict.
        """
        images = kwargs.get("images", kwargs.get("batch_images"))
        instructions = kwargs.get("instructions")
        state = kwargs.get("state")

        # 1. Encode
        backbone_output = self.encode(images, instructions)

        # 2. condition
        aligned = self.condition.inject(backbone_output)

        # 3. Predict
        action_context = aligned["action_context"]
        pred_actions = self.action_loss.predict(
            action_context=action_context,
            state=state,
        )

        return {"normalized_actions": pred_actions.cpu().numpy()}
