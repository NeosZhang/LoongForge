# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
PaliGemmaPi05 - PiVLA PI05Pytorch wrapped in LoongForgeVLA architecture interface.

Reuses PiVLA's full model (PaliGemma VLM + Gemma action expert + joint attention),
adapts LoongForgeVLA dataloader output format to PI05Pytorch's input interface.

Dataloader provides:
    {"image": [PIL.Image, ...], "lang": str, "action": ndarray[T, D], "state": ndarray|None}

This architecture handles:
    1. PIL Image -> tensor conversion + resize
    2. State discretization + prompt construction (pi0.5)
    3. Tokenization
    4. Action padding
    5. Delegates to PI05Pytorch.forward() for joint-attention flow matching
"""

import logging
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loongforge.embodied.model.compose.model_registry import ARCHITECTURE_REGISTRY
from loongforge.embodied.model.compose.architecture.base import BaseArchitecture
from loongforge.embodied.model.compose.condition.base import BaseCondition
from loongforge.embodied.model.compose.action.base import BaseAction
from loongforge.embodied.model.modules.pi05 import (
    PI05Config,
    PI05Pytorch,
    prepare_batch_state_prompts,
    tokenize_prompts,
    build_tokenizer,
)
from loongforge.embodied.train.global_vars import get_args

logger = logging.getLogger(__name__)


def _pil_to_tensor(images_pil: list, image_size: int, device: torch.device) -> torch.Tensor:
    """Convert list of PIL Images to (B, 3, H, W) float32 tensor."""
    from torchvision.transforms.functional import to_tensor, resize

    tensors = []
    for img in images_pil:
        if not isinstance(img, torch.Tensor):
            t = to_tensor(img)  # (3, H, W) float [0,1]
        else:
            t = img.float()
            if t.max() > 1.0:
                t = t / 255.0
        if t.shape[-2] != image_size or t.shape[-1] != image_size:
            t = resize(t, [image_size, image_size])
        tensors.append(t)
    return torch.stack(tensors).to(device)


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

    def forward(self, examples: List[Dict[str, Any]], **kwargs) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            examples: list of dicts with keys: image, lang, action, (state)

        Returns:
            {"action_loss": scalar, "flow_matching_loss": float}
        """
        device = next(self.parameters()).device
        B = len(examples)

        # 1. Process images: list of (B, 3, H, W) tensors, one per view
        images_list = []
        img_masks = []
        for view_idx in range(self.num_images):
            view_images = []
            for ex in examples:
                img_list = ex["image"]
                if view_idx < len(img_list):
                    view_images.append(img_list[view_idx])
                else:
                    # Pad with black image if view missing
                    view_images.append(torch.zeros(3, self.image_size, self.image_size))
            images_list.append(_pil_to_tensor(view_images, self.image_size, device))
            mask_val = self.image_mask[view_idx] if view_idx < len(self.image_mask) else False
            img_masks.append(
                torch.full((B,), mask_val, dtype=torch.bool, device=device)
            )

        # 2. Build prompts: if discrete_state_input is handled by dataloader,
        #    lang already contains the full prompt; otherwise build it here.
        instructions = [ex["lang"] for ex in examples]
        if self.pi05 and "state" in examples[0] and examples[0]["state"] is not None:
            # Check if dataloader already processed (prompt starts with "Task:")
            if not instructions[0].startswith("Task:"):
                states = torch.stack([
                    torch.as_tensor(ex["state"], dtype=torch.float32).flatten()
                    for ex in examples
                ])
                prompts = prepare_batch_state_prompts(
                    states, instructions, max_state_dim=self.max_state_dim
                )
            else:
                prompts = instructions
        else:
            if not instructions[0].startswith("Task:"):
                prompts = [f"Task: {lang.strip()};\nAction: " for lang in instructions]
            else:
                prompts = instructions

        # 3. Tokenize
        tok_out = tokenize_prompts(
            prompts, self.tokenizer, max_length=self.max_token_len
        )
        tokens = tok_out["input_ids"].to(device)
        masks = tok_out["attention_mask"].bool().to(device)

        # 4. Process actions: pad to max_action_dim, truncate/pad to action_horizon
        actions = torch.stack([
            torch.as_tensor(ex["action"], dtype=torch.float32) for ex in examples
        ]).to(device)  # (B, T, D)

        if actions.shape[-1] < self.max_action_dim:
            actions = F.pad(actions, (0, self.max_action_dim - actions.shape[-1]))
        elif actions.shape[-1] > self.max_action_dim:
            actions = actions[..., :self.max_action_dim]

        if actions.shape[1] > self.action_horizon:
            actions = actions[:, :self.action_horizon]
        elif actions.shape[1] < self.action_horizon:
            pad = torch.zeros(
                B, self.action_horizon - actions.shape[1], actions.shape[2],
                device=device, dtype=actions.dtype
            )
            actions = torch.cat([actions, pad], dim=1)

        # 5. Forward through PI05Pytorch
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

        Returns:
            {"normalized_actions": ndarray (B, action_horizon, max_action_dim)}
        """
        device = next(self.parameters()).device
        images_raw = kwargs.get("images", kwargs.get("batch_images"))
        instructions = kwargs.get("instructions", [])
        state = kwargs.get("state", None)

        if not isinstance(images_raw[0], list):
            images_raw = [[img] for img in images_raw]

        B = len(images_raw)

        # Process images
        images_list = []
        img_masks = []
        for view_idx in range(self.num_images):
            view_images = []
            for img_list in images_raw:
                if view_idx < len(img_list):
                    view_images.append(img_list[view_idx])
                else:
                    view_images.append(torch.zeros(3, self.image_size, self.image_size))
            images_list.append(_pil_to_tensor(view_images, self.image_size, device))
            mask_val = self.image_mask[view_idx] if view_idx < len(self.image_mask) else False
            img_masks.append(
                torch.full((B,), mask_val, dtype=torch.bool, device=device)
            )

        # Build prompts
        if self.pi05 and state is not None:
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            prompts = prepare_batch_state_prompts(
                state, instructions, max_state_dim=self.max_state_dim
            )
        else:
            prompts = [f"Task: {lang.strip()};\nAction: " for lang in instructions]

        # Tokenize
        tok_out = tokenize_prompts(
            prompts, self.tokenizer, max_length=self.max_token_len
        )
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

