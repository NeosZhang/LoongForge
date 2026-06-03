# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pi05 model-specific data transforms and collator.

Per-sample transform:
    StateDiscretizationTransform: state → normalize → discretize → embed in prompt

Batch-level collator (DataLoader collate_fn):
    @register_preprocessor("PaliGemmaPi05")
    Pi05Preprocessor: transformed samples → Pi05PreparedBatch (CPU tensors)

Utilities:
    tokenize_prompts(prompts, tokenizer, max_length) -> dict
    build_tokenizer(config) -> AutoTokenizer
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from loongforge.embodied.data.transforms.base import BaseTransform
from loongforge.embodied.data.transforms.normalizer import Normalizer
from loongforge.embodied.model.modules.pi05.configuration_pi05 import PI05Config
from loongforge.embodied.data.transforms.pipeline import BasePreprocessor, PreparedBatch, register_preprocessor


class StateDiscretizationTransform(BaseTransform):
    """Discretize state and embed into language prompt.

    Converts:
        state (raw) -> normalize -> discretize (N bins, no clip) -> embed in prompt
    """

    def __init__(
        self,
        apply_to: List[str] = None,
        state_key: str = "observation.state",
        task_key: str = "lang",
        num_bins: int = 256,
        max_state_dim: Optional[int] = None,
        prompt_template: str = "Task: {task}, State: {state};\nAction: ",
        normalization_mode: str = "q99",
        statistics: Optional[Dict[str, Any]] = None,
        training: bool = True,
    ):
        super().__init__(apply_to=apply_to or ["lang"], training=training)
        self.state_key = state_key
        self.task_key = task_key
        self.num_bins = num_bins
        self.max_state_dim = max_state_dim
        self.prompt_template = prompt_template

        self.normalizer = None
        if statistics is not None:
            self.normalizer = Normalizer(
                mode=normalization_mode,
                statistics=statistics,
            )

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Discretize state and embed into language prompt.

        Args:
            data: Dict containing state_key and task_key fields.

        Returns:
            Updated dict with discretized state embedded in prompt at apply_to[0].
        """
        state = data.get(self.state_key)
        task = data.get(self.task_key, "perform the task")

        if state is None:
            data[self.apply_to[0]] = f"Task: {task.strip()};\nAction: "
            return data

        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.float().flatten()

        if self.normalizer is not None:
            state = self.normalizer.forward(state)

        actual_dim = state.shape[0]
        effective_dim = self.max_state_dim if self.max_state_dim else actual_dim

        if actual_dim < effective_dim:
            state = torch.nn.functional.pad(state, (0, effective_dim - actual_dim))
        elif actual_dim > effective_dim:
            state = state[:effective_dim]

        state_np = state.cpu().numpy()
        bins = np.linspace(-1, 1, self.num_bins + 1)[:-1]
        discretized = np.digitize(state_np, bins=bins) - 1

        cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized))
        data[self.apply_to[0]] = self.prompt_template.format(
            task=cleaned_text, state=state_str
        )

        return data

def tokenize_prompts(
    prompts: list,
    tokenizer,
    max_length: int = 200,
    padding: str = "max_length",
    padding_side: str = "right",
) -> dict:
    """Tokenize a list of prompts with the PaliGemma tokenizer."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            max_length=max_length,
            padding=padding,
            truncation=True,
        )
    finally:
        tokenizer.padding_side = original_padding_side
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def build_tokenizer(config: PI05Config):
    """Build the PaliGemma tokenizer from PI05Config."""
    if not config.tokenizer_name:
        raise ValueError(
            "PI05Config.tokenizer_name is empty. "
            "Set it to the tokenizer path or set TOKENIZER_PATH env variable."
        )
    return AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        local_files_only=config.tokenizer_local_files_only,
    )


@dataclass
class Pi05PreparedBatch(PreparedBatch):
    """Preprocessed batch for Pi05 model.

    All tensors on CPU after collation; call .to(device) before forward().
    """
    images_list: List[torch.Tensor] = None   # List of (B, 3, H, W) per view
    img_masks: List[torch.Tensor] = None     # List of (B,) bool per view
    input_ids: torch.Tensor = None           # (B, seq_len)
    attention_mask: torch.Tensor = None      # (B, seq_len) bool
    actions: torch.Tensor = None             # (B, T, D)


@register_preprocessor("PaliGemmaPi05")
class Pi05Preprocessor(BasePreprocessor):
    """DataLoader collate_fn for PaliGemmaPi05.

    Only handles batch-level collation. Per-sample transforms (image, action,
    state discretization) are applied via the injected `transform` pipeline.
    """

    def __init__(
        self,
        image_size: int = 224,
        num_images: int = 2,
        image_mask: Optional[List[bool]] = None,
        max_token_len: int = 200,
        tokenizer_path: str = "",
        transform=None,
    ):
        self.image_size = image_size
        self.num_images = num_images
        self.image_mask = image_mask or [True] * num_images
        self.max_token_len = max_token_len
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None
        self.transform = transform

    @classmethod
    def from_config(cls, cfg) -> "Pi05Preprocessor":
        """Construct from full training config (OmegaConf)."""
        fw_cfg = getattr(cfg, "framework", None)
        backbone_cfg = fw_cfg.get("backbone", {}) if fw_cfg else {}

        tokenizer_path = backbone_cfg.get("tokenizer_name", "") or os.environ.get("TOKENIZER_PATH", "")

        return cls(
            image_size=backbone_cfg.get("image_size", 224),
            num_images=backbone_cfg.get("num_images", 2),
            image_mask=backbone_cfg.get("image_mask", None),
            max_token_len=backbone_cfg.get("max_token_len", 200),
            tokenizer_path=tokenizer_path,
        )

    @property
    def tokenizer(self):
        """Lazy-loaded tokenizer (loaded once per DataLoader worker)."""
        if self._tokenizer is None:
            path = self.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
            if not path:
                raise ValueError(
                    "Tokenizer path not set. Pass tokenizer_path or set TOKENIZER_PATH env."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        return self._tokenizer

    def __call__(self, examples: List[Dict[str, Any]]) -> Pi05PreparedBatch:
        """Apply per-sample transforms then collate into Pi05PreparedBatch."""
        if self.transform is not None:
            examples = [self.transform(ex) for ex in examples]

        B = len(examples)
        images_list, img_masks = self._collate_images(examples, B)
        prompts = [ex.get("prompt", self._fallback_prompt(ex)) for ex in examples]
        input_ids, attention_mask = self._tokenize(prompts)

        actions = torch.stack([
            ex["action"] if isinstance(ex["action"], torch.Tensor)
            else torch.as_tensor(ex["action"], dtype=torch.float32)
            for ex in examples
        ])

        return Pi05PreparedBatch(
            images_list=images_list,
            img_masks=img_masks,
            input_ids=input_ids,
            attention_mask=attention_mask,
            actions=actions,
        )

    def _collate_images(self, examples, B):
        """Stack pre-processed images per view."""
        image_keys = sorted(
            k for k in examples[0].keys() if k.startswith("observation.images.")
        )

        images_list = []
        img_masks = []
        for view_idx in range(self.num_images):
            view_images = []
            for ex in examples:
                if view_idx < len(image_keys):
                    img = ex[image_keys[view_idx]]
                    if not isinstance(img, torch.Tensor):
                        img = torch.as_tensor(img, dtype=torch.float32)
                    view_images.append(img)
                else:
                    view_images.append(torch.zeros(3, self.image_size, self.image_size) - 1.0)
            images_list.append(torch.stack(view_images))
            mask_val = self.image_mask[view_idx] if view_idx < len(self.image_mask) else False
            img_masks.append(torch.full((B,), mask_val, dtype=torch.bool))
        return images_list, img_masks

    def _fallback_prompt(self, ex):
        """Build a simple prompt when StateDiscretizationTransform was not applied."""
        task = ex.get("task", "perform the task")
        return f"Task: {task.strip()};\nAction: "

    def _tokenize(self, prompts):
        tok_out = tokenize_prompts(
            prompts, self.tokenizer, max_length=self.max_token_len
        )
        return tok_out["input_ids"], tok_out["attention_mask"].bool()
