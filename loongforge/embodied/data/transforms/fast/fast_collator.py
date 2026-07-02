# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fast batch-level collator (DataLoader collate_fn).

Performs the full data preprocessing pipeline that was previously inside
QwenFast.forward():
  1. FAST tokenization (action ndarray → discrete tokens)
  2. Token mapping (FAST tokens → <robot_action_N> strings)
  3. QwenVL tokenization (images + text + action tokens → BatchFeature)
  4. Label masking (mask non-action tokens with -100)

After this collator, QwenFast.forward() receives ready-to-consume tensors,
mirroring how PaliGemmaPi05 receives Pi05PreparedBatch.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from loongforge.embodied.data.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
)

_IGNORE_INDEX = -100


@dataclass
class FastPreparedBatch(PreparedBatch):
    """Preprocessed batch for QwenFast model.

    All tensors on CPU after collation; call .to(device) before forward().
    """
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None
    pixel_values: torch.Tensor = None
    image_grid_thw: torch.Tensor = None
    labels: torch.Tensor = None


@register_preprocessor("Fast")
class FastPreprocessor(BasePreprocessor):
    """DataLoader collate_fn for QwenFast.

    Moves FAST tokenization + QwenVL processing out of model.forward()
    into data preprocessing (same pattern as Pi05Preprocessor).
    """

    def __init__(self, fast_tokenizer_path: str = "", vlm_path: str = "", config: Optional[dict] = None):
        self.fast_tokenizer_path = fast_tokenizer_path
        self.vlm_path = vlm_path
        self.config = config or {}
        self._fast_tokenizer = None
        self._vlm_processor = None

    @classmethod
    def from_config(cls, cfg, args=None) -> "FastPreprocessor":
        """Build a FastPreprocessor from model config."""
        backbone_cfg = cfg.get("backbone", {}) if hasattr(cfg, "get") else {}
        vlm_path = backbone_cfg.get("base_vlm", "")
        # FAST tokenizer path: from TOKENIZER_PATH env (points to physical-intelligence/fast model)
        fast_tokenizer_path = os.environ.get("TOKENIZER_PATH", "physical-intelligence/fast")
        return cls(fast_tokenizer_path=fast_tokenizer_path, vlm_path=vlm_path, config=cfg)

    @property
    def fast_tokenizer(self):
        """Lazy-loaded FAST action tokenizer."""
        if self._fast_tokenizer is None:
            from loongforge.embodied.model.compose.action.fast_action import _load_fast_processor
            self._fast_tokenizer = _load_fast_processor(self.fast_tokenizer_path)
        return self._fast_tokenizer

    @property
    def vlm_processor(self):
        """Lazy-loaded Qwen VLM processor."""
        if self._vlm_processor is None:
            from transformers import AutoProcessor
            self._vlm_processor = AutoProcessor.from_pretrained(self.vlm_path)
            self._vlm_processor.tokenizer.padding_side = "left"
        return self._vlm_processor

    def __call__(self, examples: List[Dict[str, Any]]) -> FastPreparedBatch:
        """Collate: FAST tokenize → build QwenVL inputs → label masking."""
        from qwen_vl_utils import process_vision_info
        from loongforge.embodied.model.compose.action.fast_action import (
            _ACTION_TOKEN_MIN,
            _ACTION_TOKEN_MAX,
        )

        images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]

        # Step 1: FAST tokenize actions → token sequences
        batch_actions = np.stack(actions, axis=0)
        batch_fast_tokens = self.fast_tokenizer(batch_actions)

        # Step 2: Map FAST tokens to VLM action token strings
        vlm_action_tokens = [
            tokens if isinstance(tokens, str) else "".join(f"<robot_action_{t}>" for t in tokens)
            for tokens in batch_fast_tokens
        ]

        # Step 3: Build QwenVL inputs
        backbone_cfg = self.config.get("backbone", {}) if hasattr(self.config, "get") else {}
        cot_prompt = backbone_cfg.get("CoT_prompt", None)

        messages = []
        for imgs, instruction, solution in zip(images, instructions, vlm_action_tokens):
            content = [{"type": "image", "image": img} for img in imgs]
            prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt else instruction
            content.append({"type": "text", "text": prompt})
            msg = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": solution}]},
            ]
            messages.append(msg)

        texts = [
            self.vlm_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.vlm_processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )

        # Step 4: Build labels (mask non-action tokens)
        labels = batch_input["input_ids"].clone()
        for i in range(labels.size(0)):
            seq = labels[i]
            mask_seq = (seq >= _ACTION_TOKEN_MIN) & (seq <= _ACTION_TOKEN_MAX)
            nonzero = torch.nonzero(mask_seq, as_tuple=False)
            if nonzero.numel() > 0:
                seq[:nonzero[0].item()] = _IGNORE_INDEX
            else:
                seq[:] = _IGNORE_INDEX
        labels[labels == self.vlm_processor.tokenizer.pad_token_id] = _IGNORE_INDEX

        return FastPreparedBatch(
            input_ids=batch_input["input_ids"],
            attention_mask=batch_input["attention_mask"],
            pixel_values=batch_input.get("pixel_values"),
            image_grid_thw=batch_input.get("image_grid_thw"),
            labels=labels,
        )
