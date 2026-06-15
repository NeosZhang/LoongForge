# loongforge/embodied/model/compose/architecture/QwenFast.py

"""
QwenFast - Qwen2.5-VL + FAST Tokenizer Architecture

End-to-end autoregressive discrete action prediction architecture:
  - Qwen2.5-VL visual-language backbone
  - FAST tokenizer discrete action encoding
  - Predicts action sequences via next-token prediction
"""

from typing import Dict, List, Any
import logging
import numpy as np
import torch
import torch.nn as nn

from loongforge.embodied.model.compose.model_registry import ARCHITECTURE_REGISTRY
from loongforge.embodied.model.compose.architecture.base import BaseArchitecture
from loongforge.embodied.model.compose.condition.base import BaseCondition
from loongforge.embodied.model.compose.action.base import BaseAction
from loongforge.embodied.model.modules.vlm import get_vlm_model
from loongforge.embodied.model.compose.action.fast_action import get_action_model as get_fast_action_tokenizer
from loongforge.embodied.train.global_vars import get_args

logger = logging.getLogger(__name__)


@ARCHITECTURE_REGISTRY.register("QwenFast")
class QwenFast(BaseArchitecture):
    """
    QwenFast architecture: Qwen VLM + FAST tokenizer end-to-end action prediction.

    Data flow:
      images + instructions
            |
            v
      ┌──────────────┐
      │  Qwen VL     │  (Qwen2.5-VL / Qwen3-VL)
      │   backbone   │
      └──────┬───────┘
             │ hidden_states
             v
      ┌──────────────┐
      │ FAST Tokenizer│  (discrete action token encoding/decoding)
      └──────┬───────┘
             │ action_tokens / actions
             v
    """

    def __init__(self, config, condition: BaseCondition, action: BaseAction):
        """
        Args:
            config: Complete framework config (includes backbone, action_model, etc.)
            condition: Must be None (QwenFast is an end-to-end architecture)
            action: Must be None (FAST tokenizer is built-in)
        """
        super().__init__(config, condition, action)

        # Validate condition and action are None (end-to-end architecture)
        if condition is not None or action is not None:
            logger.warning(
                "QwenFast is an end-to-end architecture; "
                "condition and action parameters will be ignored."
            )

        # Extract config
        backbone_cfg = config.get("backbone", {})
        action_cfg = config.get("action_model", {})

        self.action_dim = action_cfg.get("action_dim", 7)
        self.action_horizon = action_cfg.get("action_horizon", 50)

        # Core modules
        self._qwen_vl = get_vlm_model(config)  # VLM interface
        self._fast_tokenizer = get_fast_action_tokenizer(config)  # FAST tokenizer

        # Action token range
        self._ACTION_TOKEN_MIN = getattr(self._qwen_vl, "_ACTION_TOKEN_MIN", 151665)
        self._ACTION_TOKEN_MAX = getattr(self._qwen_vl, "_ACTION_TOKEN_MAX", 153712)

    @property
    def backbone(self) -> nn.Module:
        """Return Qwen VL model"""
        return self._qwen_vl.model

    @property
    def action_head(self) -> nn.Module:
        """Return FAST tokenizer as action head"""
        return self._fast_tokenizer

    def encode(self, images, instructions, **kwargs):
        """
        Encode images and instructions (QwenFast doesn't use encode separately, goes through forward directly)
        """
        raise NotImplementedError(
            "QwenFast uses end-to-end forward(). "
            "Use forward() or predict_action() directly."
        )

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            batch: FastPreparedBatch from dataloader preprocessor, must have attributes:
                   input_ids, attention_mask, pixel_values, image_grid_thw, labels
                   All tensors should already be on the correct device (via batch.to(device)).

        Returns:
            {"action_loss": scalar tensor}
        """
        assert hasattr(batch, "input_ids"), (
            "forward() expects a FastPreparedBatch from FastPreprocessor, "
            "got raw data instead. Ensure DataLoader uses the registered preprocessor as collate_fn."
        )

        # Build kwargs for VLM forward
        forward_kwargs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "labels": batch.labels,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if batch.pixel_values is not None:
            forward_kwargs["pixel_values"] = batch.pixel_values
        if batch.image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = batch.image_grid_thw

        # VLM forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self._qwen_vl(**forward_kwargs)

        # Extract loss
        action_loss = outputs.loss
        if action_loss is None or torch.isnan(action_loss):
            action_loss = torch.tensor(0.0, device=batch.input_ids.device)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Inference: Generate action sequences

        Args:
            images: list of PIL Images (multi-view)
            instructions: list of str

        Returns:
            {"normalized_actions": ndarray (B, T, action_dim)}
        """
        images_raw = kwargs.get("images", kwargs.get("batch_images"))
        instructions = kwargs.get("instructions", [])

        if not isinstance(images_raw[0], list):
            images_raw = [[img] for img in images_raw]

        # Step 1: Build inputs (without solution)
        qwen_inputs = self._qwen_vl.build_qwenvl_inputs(
            images=images_raw,
            instructions=instructions
        )

        # Step 2: Generation
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self._qwen_vl.model.generate(
                **qwen_inputs,
                max_length=2048,
            )

        # Step 3: Extract action token ids
        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)

        # Step 4: Decode to FAST token ids
        batch_fast_token_ids = self._decode_action_tokens(batch_vlm_action_token_ids)

        # Step 5: Decode FAST tokens to actions
        normalized_actions = self._fast_tokenizer.fast_tokenizer.decode(batch_fast_token_ids)

        return {"normalized_actions": normalized_actions}

    def _extract_action_token_ids(self, generated_ids: torch.LongTensor) -> List[List[int]]:
        """Extract action tokens from generated sequences"""
        mask = (generated_ids >= self._ACTION_TOKEN_MIN) & (generated_ids <= self._ACTION_TOKEN_MAX)
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
            else:
                results.append(generated_ids[b, idx].tolist())
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """Decode VLM action tokens to FAST token ids"""
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
            else:
                batch_fast_token_ids.append([t - self._ACTION_TOKEN_MIN for t in seq])
        return batch_fast_token_ids

    def _map_fast_token_to_vlm_action(self, tokens) -> str:
        """Map FAST tokens to VLM action format"""
        return "".join([f"<robot_action_{token}>" for token in tokens])

    def load_pretrained(self, checkpoint_path: str, **kwargs):
        """Load fine-tuned checkpoint weights using load_state_dict."""
        import torch
        
        device = kwargs.get("device", next(self.parameters()).device)
        
        # Load checkpoint
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=str(device))
        
        # Load state dict
        self.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        if hasattr(self._qwen_vl.model, "gradient_checkpointing_enable"):
            self._qwen_vl.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self._qwen_vl.model, "gradient_checkpointing_disable"):
            self._qwen_vl.model.gradient_checkpointing_disable()
