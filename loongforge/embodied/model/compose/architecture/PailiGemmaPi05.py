# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
PaliGemmaPi05 - PiVLA PI05Pytorch wrapped in LoongForgeVLA architecture interface.

Reuses PiVLA's full model (PaliGemma VLM + Gemma action expert + joint attention).

DataLoader provides Pi05PreparedBatch (from Pi05Preprocessor collate_fn):
    .images_list: List[Tensor (B, 3, H, W)] in [-1, 1]
    .img_masks:   List[Tensor (B,) bool]
    .input_ids:   Tensor (B, seq_len)
    .attention_mask: Tensor (B, seq_len) bool
    .actions:     Tensor (B, T, D)

This architecture simply delegates to PI05Pytorch.forward() for joint-attention flow matching.
"""

import logging
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from loongforge.embodied.model.compose.model_registry import ARCHITECTURE_REGISTRY
from loongforge.embodied.model.compose.architecture.base import BaseArchitecture
from loongforge.embodied.model.compose.condition.base import BaseCondition
from loongforge.embodied.model.compose.action.base import BaseAction
from loongforge.embodied.model.modules.pi05 import (
    PI05Config,
    PI05Pytorch,
    tokenize_prompts,
    build_tokenizer,
)
from loongforge.embodied.data.transforms import ImageTransform, StateDiscretizationTransform
from loongforge.embodied.data.transforms.pipeline import convert_stats
from loongforge.embodied.train.global_vars import get_args

logger = logging.getLogger(__name__)


@ARCHITECTURE_REGISTRY.register("PaliGemmaPi05")
class PaliGemmaPi05(BaseArchitecture):
    """PiVLA PI05Pytorch model wrapped as LoongForgeVLA architecture.

    Bypasses the standard VLMActionModel encode->condition->action flow,
    because PI05Pytorch internally manages its own VLM + action expert
    with joint attention.
    """

    def __init__(self, config, condition: BaseCondition, action: BaseAction):
        super().__init__(config, condition, action)

        # Extract config values (backbone/action_model are top-level keys in config)
        backbone_cfg = config.get("backbone", {})
        action_cfg = config.get("action_model", {})

        self.pi05 = config.get("pi05", True)
        self.image_size = backbone_cfg.get("image_size", 224)
        self.num_images = backbone_cfg.get("num_images", 2)
        self.image_mask = backbone_cfg.get("image_mask", [True] * self.num_images)
        self.max_token_len = backbone_cfg.get("max_token_len", 48)

        self.action_dim = action_cfg.get("action_dim", 7)
        self.action_horizon = action_cfg.get("action_horizon", 10)
        self.max_action_dim = action_cfg.get("max_action_dim", 32)
        self.max_state_dim = action_cfg.get("max_state_dim", 32)

        # Build PI05Config
        paligemma_variant = backbone_cfg.get("paligemma_variant", "gemma_2b")
        action_expert_variant = action_cfg.get("action_expert_variant", "gemma_300m")
        precision = action_cfg.get("precision", "bfloat16")
        args = get_args()
        tokenizer_name = getattr(args, "tokenizer_path", None)
        assert tokenizer_name, "tokenizer_path must be provided via --tokenizer-path"

        self.pi05_config = PI05Config(
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            dtype=precision,
            chunk_size=self.action_horizon,
            n_action_steps=self.action_horizon,
            max_action_dim=self.max_action_dim,
            max_state_dim=self.max_state_dim,
            image_resolution=(self.image_size, self.image_size),
            tokenizer_name=tokenizer_name,
            tokenizer_max_length=self.max_token_len,
            freeze_vision_encoder=getattr(args, "freeze_vision_encoder", False),
            train_expert_only=getattr(args, "train_expert_only", False),
            gradient_checkpointing=config.get("gradient_checkpointing", False),
        )

        # Core PI05 model from PiVLA
        self.pi05_model = PI05Pytorch(self.pi05_config)

        # Tokenizer (lazy init)
        self._tokenizer = None

        # Image transform for inference (same as training pipeline)
        self._image_transform = ImageTransform(apply_to=[], image_size=self.image_size)

    @property
    def tokenizer(self):
        """Lazy tokenizer build."""
        if self._tokenizer is None:
            self._tokenizer = build_tokenizer(self.pi05_config)
        return self._tokenizer

    @property
    def backbone(self) -> nn.Module:
        """PI05Pytorch backbone (shared with action_head)."""
        return self.pi05_model.paligemma_with_expert

    @property
    def action_head(self) -> nn.Module:
        """PI05Pytorch is the action head itself (contains expert internally)."""
        return self.pi05_model

    def load_pretrained(self, checkpoint_path: str, device=None):
        """Load HuggingFace-format pi05 checkpoint into pi05_model."""
        self.pi05_model.load_pretrained(checkpoint_path, strict=False, device=device)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.pi05_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.pi05_model.gradient_checkpointing_disable()

    def encode(self, images, instructions, **kwargs):
        """PI05Pytorch encode() is not directly usable."""
        raise NotImplementedError(
            "PaliGemmaPi05 uses PI05Pytorch's internal encode. "
            "Use forward() or predict_action() directly."
        )

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            batch: Pi05PreparedBatch from dataloader preprocessor, must have attributes:
                   images_list, img_masks, input_ids, attention_mask, actions
                   All tensors should already be on the correct device (via batch.to(device)).

        Returns:
            {"action_loss": scalar, "flow_matching_loss": float}
        """
        assert hasattr(batch, "images_list"), (
            "forward() expects a Pi05PreparedBatch from Pi05Preprocessor, "
            "got raw data instead. Ensure DataLoader uses the registered preprocessor as collate_fn."
        )

        images_list = batch.images_list
        img_masks = batch.img_masks
        tokens = batch.input_ids
        masks = batch.attention_mask
        actions = batch.actions

        # Forward through PI05Pytorch
        loss_map = self.pi05_model(images_list, img_masks, tokens, masks, actions)
        loss_mean = loss_map[:, :, :self.action_dim].mean()

        return {"action_loss": loss_mean, "flow_matching_loss": loss_mean.detach().item()}

    @torch.no_grad()
    def predict_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """Inference: generate actions via Euler ODE denoising.

        Args (via kwargs):
            images: list of PIL Images (multi-view)
            instructions: list of str
            state: optional (B, D) tensor or ndarray
            dataset_stats: optional dict of dataset statistics for state normalization

        Returns:
            {"normalized_actions": ndarray (B, action_horizon, max_action_dim)}
        """
        device = next(self.parameters()).device
        images_raw = kwargs.get("images", kwargs.get("batch_images"))
        instructions = kwargs.get("instructions", [])
        state = kwargs.get("state", None)
        dataset_stats = kwargs.get("dataset_stats", None)

        if not isinstance(images_raw[0], list):
            images_raw = [[img] for img in images_raw]

        B = len(images_raw)

        # Process images using ImageTransform (same as training)
        images_list = []
        img_masks = []
        for view_idx in range(self.num_images):
            view_images = [
                img_list[view_idx] if view_idx < len(img_list)
                else torch.zeros(3, self.image_size, self.image_size)
                for img_list in images_raw
            ]
            images_list.append(self._image_transform.process_batch(view_images).to(device))
            mask_val = self.image_mask[view_idx] if view_idx < len(self.image_mask) else False
            img_masks.append(torch.full((B,), mask_val, dtype=torch.bool, device=device))

        # Build prompts using StateDiscretizationTransform (same as training)
        if self.pi05 and state is not None:
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state_stats = convert_stats(dataset_stats.get("observation.state")) if dataset_stats else None
            state_transform = StateDiscretizationTransform(
                apply_to=["prompt"],
                state_key="observation.state",
                task_key="task",
                num_bins=256,
                max_state_dim=None,
                normalization_mode="q99",
                statistics=state_stats,
            )
            prompts = []
            for i in range(B):
                sample = {"observation.state": state[i], "task": instructions[i]}
                result = state_transform.apply(sample)
                prompts.append(result["prompt"])
        else:
            prompts = [f"Task: {lang.strip()};\nAction: " for lang in instructions]

        # Tokenize
        tok_out = tokenize_prompts(prompts, self.tokenizer, max_length=self.max_token_len)
        tokens = tok_out["input_ids"].to(device)
        masks = tok_out["attention_mask"].bool().to(device)

        # Sample actions
        pred_actions = self.pi05_model.sample_actions(
            images_list, img_masks, tokens, masks
        )
        return {"normalized_actions": pred_actions.cpu().numpy()}

    def load_pretrained(self, pretrained_path: str, strict: bool = False, **kwargs):
        """Load pretrained PI05 weights."""
        self.pi05_model.load_pretrained(pretrained_path, strict=strict)
        return self
