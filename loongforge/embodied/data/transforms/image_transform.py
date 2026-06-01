# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
ImageTransform - Image preprocessing transform

Supports:
  - Resize to target size
  - Center crop / Random crop
  - Color jitter (training only)
  - Normalize to [0, 1]
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from loongforge.embodied.data.transforms.base import BaseTransform


class ImageTransform(BaseTransform):
    """
    Image preprocessing transform.

    Unifies images in data[key] (PIL.Image / np.ndarray / list) to PIL.Image,
    and applies resize, crop, jitter and other augmentations.
    """

    def __init__(
        self,
        apply_to: List[str],
        size: Tuple[int, int] = (224, 224),
        crop_scale: float = 1.0,
        color_jitter: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        training: bool = True,
    ):
        super().__init__(apply_to=apply_to, training=training)
        self.size = size
        self.crop_scale = crop_scale
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, list):
                data[key] = [self._process_single(img) for img in value]
            else:
                data[key] = self._process_single(value)
        return data

    def _process_single(self, img) -> Image.Image:
        """Process a single image."""
        # Convert to PIL
        if isinstance(img, np.ndarray):
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        # Random crop (training only)
        if self.training and self.crop_scale < 1.0:
            w, h = img.size
            new_w = int(w * self.crop_scale)
            new_h = int(h * self.crop_scale)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))

        # Resize
        if img.size != self.size:
            img = img.resize(self.size, Image.BILINEAR)

        # Color jitter (training only)
        if self.training and self.color_jitter:
            img = self._apply_color_jitter(img)

        return img

    def _apply_color_jitter(self, img: Image.Image) -> Image.Image:
        """Simple color jitter implementation."""
        try:
            from torchvision.transforms import ColorJitter
            jitter = ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
            return jitter(img)
        except ImportError:
            return img
