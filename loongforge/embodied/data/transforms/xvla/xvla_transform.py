# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from X-VLA (https://github.com/2toinf/X-VLA).
# Copyright 2025 2toINF. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XVLA per-sample transforms.

These transforms replace :mod:`loongforge.embodied.model.xvla.processing_xvla`
and mirror the pi05_transform pattern so the xvla processing pipeline can be
assembled via ``transforms.append(...)`` in pipeline.py, exactly like Pi05.

Two transforms are provided:

* :class:`XVLATokenizeTransform`  — tokenizes ``task``/``prompt`` into
  ``input_ids`` using a BART tokenizer loaded via ``from_pretrained``.
  Mirrors :class:`Pi05TokenizeTransform`.

* :class:`XVLAEncodeImageTransform` — converts ``observation.images.*``
  tensors into the ``image_input`` / ``image_mask`` tensors expected by the
  model, using an AutoImageProcessor loaded via ``from_pretrained``.
  Mirrors :class:`Pi05CollateImagesTransform`.

Both classes lazy-load their sub-processors so that the heavy
``from_pretrained`` call happens once per DataLoader worker on first access,
not at pipeline-build time.

The image / tokenizer sub-processor paths are read from the same checkpoint
directory that was previously passed to ``XVLAProcessor.from_pretrained``.
"""

import os
from typing import Any, Dict, List, Optional, Union

import torch

from loongforge.embodied.data.transforms.base import BaseTransform
from loongforge.embodied.data.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)


class XVLATokenizeTransform(BaseTransform):
    """Tokenize language instruction into ``input_ids``.

    Loads a BART-compatible tokenizer (same tokenizer that was bundled inside
    XVLAProcessor) from the checkpoint directory and tokenizes the language
    instruction for each sample.  Writes ``input_ids`` ([L]) into the sample
    dict.

    Replaces ``XVLAProcessor.encode_language``.
    Mirrors the role of :class:`Pi05TokenizeTransform` for XVLA.
    """

    def __init__(
        self,
        tokenizer_path: str = "",
        language_max_length: int = 50,
        task_key: str = "task",
        prompt_key: str = "prompt",
        training: bool = True,
    ):
        """
        Args:
            tokenizer_path: Path to the pretrained checkpoint directory that
                            contains ``tokenizer_config.json`` (etc.).
                            Falls back to ``PROCESSOR_PATH`` / ``TOKENIZER_PATH``
                            environment variables.
            language_max_length: Maximum token length for padding/truncation.
            task_key: Fallback sample-dict key when ``prompt_key`` is absent.
            prompt_key: Primary sample-dict key for the language instruction.
            training: Whether in training mode.
        """
        super().__init__(apply_to=["input_ids"], training=training)
        self.tokenizer_path = tokenizer_path
        self.language_max_length = language_max_length
        self.task_key = task_key
        self.prompt_key = prompt_key
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer on first access (once per DataLoader worker)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

            # Ensure a pad token exists (BART/Florence2 may lack one by default)
            if tokenizer.pad_token is None:
                vocab = tokenizer.get_vocab()
                if "<pad>" in vocab:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})
                elif tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})

            self._tokenizer = tokenizer
        return self._tokenizer

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the language instruction and write ``input_ids`` into data."""
        instruction = (
            data.get(self.prompt_key)
            or data.get(self.task_key)
            or data.get("lang")
            or ""
        )
        inputs = self.tokenizer(
            [str(instruction)],
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        # Squeeze batch dim: per-sample transform produces [L], not [1, L]
        data["input_ids"] = inputs["input_ids"].squeeze(0)
        return data

    def encode_language_batch(
        self, instructions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of instructions; returns ``{"input_ids": [B, L]}``.

        Provided as a convenience for the collator path where a full batch is
        tokenized at once (matches the former ``XVLAProcessor.encode_language``
        batch interface).
        """
        inputs = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        return {"input_ids": inputs["input_ids"]}


class XVLAEncodeImageTransform(BaseTransform):
    """Encode ``observation.images.*`` views into ``image_input`` / ``image_mask``.

    Loads an AutoImageProcessor from the checkpoint directory (the same
    image_processor that was bundled inside XVLAProcessor) and applies it to
    each sample's per-view images.  Writes:

    * ``image_input``: tensor [num_views, C, H, W]
    * ``image_mask``:  bool tensor [num_views]

    into the sample dict.

    When the dataset already provides ``image_input`` (the reference-aligned
    HDF5VLADataset path), this transform is a no-op so the pipeline is
    idempotent.

    Replaces ``XVLAProcessor.encode_image``.
    Mirrors the role of :class:`Pi05CollateImagesTransform` for XVLA.
    """

    def __init__(
        self,
        tokenizer_path: str = "",
        num_views: int = 3,
        training: bool = True,
    ):
        """
        Args:
            tokenizer_path: Path to the pretrained checkpoint directory that
                            contains ``preprocessor_config.json`` (etc.).
                            Falls back to ``PROCESSOR_PATH`` / ``TOKENIZER_PATH``
                            environment variables.
            num_views: Expected number of camera views; missing views are
                       zero-padded.
            training: Whether in training mode.
        """
        super().__init__(apply_to=["image_input", "image_mask"], training=training)
        self.tokenizer_path = tokenizer_path
        self.num_views = num_views
        self._image_processor = None

    @property
    def image_processor(self):
        """Lazy-load the image processor on first access (once per DataLoader worker)."""
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            self._image_processor = AutoImageProcessor.from_pretrained(self.tokenizer_path)
        return self._image_processor

    @staticmethod
    def _to_pil(img: torch.Tensor):
        """Convert a CHW float/uint8 tensor to a PIL Image."""
        from torchvision.transforms.functional import to_pil_image

        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0
        return to_pil_image(img.clamp(0.0, 1.0))

    def _encode_single(self, pil_images: List) -> Dict[str, torch.Tensor]:
        """Encode a single sample's list of PIL images into image_input / image_mask.

        Replicates ``XVLAProcessor.encode_image`` for a single sample.
        """
        processed = self.image_processor(pil_images, return_tensors="pt")["pixel_values"]
        V_exist = processed.size(0)

        if V_exist < self.num_views:
            processed = torch.cat(
                [processed,
                 processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                dim=0,
            )

        image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=processed.device)
        image_mask[:V_exist] = True

        return {"image_input": processed, "image_mask": image_mask}

    def encode_image_batch(
        self, batch_pil_images: List[List]
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of per-sample PIL image lists; returns batched tensors.

        Provided as a convenience for the collator path where a full batch of
        raw images is encoded at once (matches the former
        ``XVLAProcessor.encode_image`` batch interface).

        Returns:
            {
              "image_input": [B, num_views, C, H, W],
              "image_mask":  [B, num_views],
            }
        """
        batch_imgs, batch_masks = [], []
        for pil_images in batch_pil_images:
            encoded = self._encode_single(pil_images)
            batch_imgs.append(encoded["image_input"])
            batch_masks.append(encoded["image_mask"])
        return {
            "image_input": torch.stack(batch_imgs, dim=0),
            "image_mask": torch.stack(batch_masks, dim=0),
        }

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encode per-view images and write ``image_input`` / ``image_mask`` into data.

        If ``image_input`` already exists (reference-aligned path), this
        transform is a no-op so the pipeline remains idempotent.
        """
        if "image_input" in data:
            return data

        image_keys = sorted(k for k in data.keys() if k.startswith("observation.images."))
        pil_images = [self._to_pil(data[k]) for k in image_keys]

        encoded = self._encode_single(pil_images)
        data["image_input"] = encoded["image_input"]   # [num_views, C, H, W]
        data["image_mask"] = encoded["image_mask"]     # [num_views]
        return data

@register_transform_builder("xvla")
def build_xvla_transforms(ctx: TransformBuilderContext):
    """Append XVLA-specific per-sample transforms if model_type is xvla.

    Two transforms are appended in order:
    1. XVLAEncodeImageTransform – encode ``observation.images.*`` views into
       ``image_input`` / ``image_mask`` via XVLAProcessor.encode_image.
       No-op when the dataset already provides ``image_input`` (reference path).
    2. XVLATokenizeTransform – tokenize the language instruction into
       ``input_ids`` via XVLAProcessor.encode_language.
    """
    transforms: list = []
    model_cfg = ctx.model_cfg
    model_type = model_cfg.model_type
    if model_type != "xvla":
        return transforms

    tokenizer_path = ctx.training_args.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
    num_views = (
        ctx.data_cfg.num_image_views
        or model_cfg.num_image_views
        or 3
    )

    transforms.append(XVLAEncodeImageTransform(
        tokenizer_path=tokenizer_path,
        num_views=num_views,
    ))
    transforms.append(XVLATokenizeTransform(
        tokenizer_path=tokenizer_path,
    ))
    return transforms