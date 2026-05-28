# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""XVLA special per-sample transforms: image collation, prompt building, tokenization, domain id.

Migrated from lerobot ``processor_xvla.py`` dataset-processing logic and adapted to
the LoongForge per-sample transform contract. Image [0,1] + ImageNet normalization is
handled upstream by the generic ``ImageTransform(normalize_mode="imagenet")``; the
steps here cover the XVLA-specific pieces that have no generic equivalent.
"""

import os
from typing import Any, Dict, List, Optional

import torch

from loongforge.embodied.data.transforms.base import BaseTransform
from loongforge.embodied.data.transforms.pi05.pi05_collator import tokenize_prompts


class XVLACollateImagesTransform(BaseTransform):
    """Collate observation images into stacked per-view tensors with masks.

    Reads keys matching ``observation.images.*``, produces:
        - "images_list": List of (3, H, W) tensors per view
        - "img_masks": List of bool values per view

    Missing views are filled with a zero placeholder and masked out.
    """

    def __init__(
        self,
        image_size: int = 224,
        num_images: int = 2,
        image_mask: Optional[List[bool]] = None,
        training: bool = True,
    ):
        super().__init__(apply_to=["images_list", "img_masks"], training=training)
        self.image_size = image_size
        self.num_images = num_images
        self.image_mask = image_mask or [True] * num_images

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collate observation images into stacked per-view tensors with masks."""
        image_keys = sorted(k for k in data.keys() if k.startswith("observation.images."))

        images_list = []
        img_masks = []
        for view_idx in range(self.num_images):
            has_view = view_idx < len(image_keys)
            if has_view:
                img = data[image_keys[view_idx]]
                if not isinstance(img, torch.Tensor):
                    img = torch.as_tensor(img, dtype=torch.float32)
            else:
                img = torch.zeros(3, self.image_size, self.image_size)
            images_list.append(img)
            mask_val = (
                has_view
                and view_idx < len(self.image_mask)
                and self.image_mask[view_idx]
            )
            img_masks.append(mask_val)

        data["images_list"] = images_list
        data["img_masks"] = img_masks
        return data


class XVLAImageNetNormalizeTransform(BaseTransform):
    """Normalize XVLA image observations in-place without resizing.

    Official XVLA/LeRobot preprocessing keeps camera tensors under their original
    observation keys and lets the policy resize with padding inside the model.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, training: bool = True):
        super().__init__(apply_to=[], training=training)
        self._mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize image observations in-place using ImageNet mean/std."""
        for key in [k for k in data.keys() if k.startswith("observation.images.")]:
            img = data[key]
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img, dtype=torch.float32)
            img = img.float()
            if img.max() > 1.0:
                img = img / 255.0
            mean = self._mean.to(device=img.device, dtype=img.dtype)
            std = self._std.to(device=img.device, dtype=img.dtype)
            data[key] = (img - mean) / std
        return data


class XVLAPromptTransform(BaseTransform):
    """Ensure a 'prompt' key exists, built from the task/lang description.

    Mirrors the XVLA inference prompt format: ``Task: {task};\\nAction: ``.
    """

    def __init__(
        self,
        task_key: str = "task",
        prompt_key: str = "prompt",
        training: bool = True,
    ):
        super().__init__(apply_to=[prompt_key], training=training)
        self.task_key = task_key
        self.prompt_key = prompt_key

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt from task/lang key if not already present."""
        if data.get(self.prompt_key) is None:
            task = data.get(self.task_key) or data.get("lang") or "perform the task"
            data[self.prompt_key] = f"Task: {task.strip()};\nAction: "
        return data


class XVLATokenizeTransform(BaseTransform):
    """Tokenize the 'prompt' field into 'input_ids' and 'attention_mask'.

    Uses an ``AutoTokenizer`` (e.g. facebook/bart-large) with configurable max length,
    matching ``TokenizerProcessorStep`` from the lerobot XVLA processor.
    """

    def __init__(
        self,
        tokenizer_path: str = "",
        max_token_len: int = 64,
        padding_side: str = "right",
        prompt_key: str = "prompt",
        training: bool = True,
    ):
        super().__init__(apply_to=["input_ids", "attention_mask"], training=training)
        self.tokenizer_path = tokenizer_path
        self.max_token_len = max_token_len
        self.padding_side = padding_side
        self.prompt_key = prompt_key
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer on first access (once per DataLoader worker)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            path = self.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
            if not path:
                raise ValueError(
                    "Tokenizer path not set. Pass tokenizer_path or set TOKENIZER_PATH env."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        return self._tokenizer

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize prompt field into input_ids and attention_mask."""
        prompt = data[self.prompt_key]
        tok_out = tokenize_prompts(
            [prompt],
            self.tokenizer,
            max_length=self.max_token_len,
            padding_side=self.padding_side,
        )
        data["observation.language.tokens"] = tok_out["input_ids"].squeeze(0)
        data["observation.language.attention_mask"] = tok_out["attention_mask"].squeeze(0).bool()
        return data


class XVLAAddDomainIdTransform(BaseTransform):
    """Attach a per-sample ``domain_id`` scalar.

    Mirrors ``XVLAAddDomainIdProcessorStep``: used by XVLA to identify different
    robot embodiments / task domains.
    """

    def __init__(
        self,
        domain_id: int = 0,
        domain_key: str = "domain_id",
        training: bool = True,
    ):
        super().__init__(apply_to=[domain_key], training=training)
        self.domain_id = domain_id
        self.domain_key = domain_key

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Attach domain_id scalar if not already present."""
        if data.get(self.domain_key) is None:
            data[self.domain_key] = torch.tensor(int(self.domain_id), dtype=torch.long)
        return data
