# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fast per-sample transform: maps LeRobot dataset output to QwenFast model input format.

Input (from LeRobotV2Dataset / LeRobotV3Dataset after ActionTransform):
    observation.images.*: Tensor [C, H, W] float32 in [0, 1]
    action: Tensor [T, D] (already normalized by ActionTransform)
    task: str

Output (for QwenFast.forward):
    image: List[PIL.Image]
    lang: str
    action: np.ndarray [T, D]
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from loongforge.embodied.data.transforms.base import BaseTransform


class FastKeyMappingTransform(BaseTransform):
    """Map standard VLA dataset output to QwenFast model input format."""

    def __init__(self, image_size: int = 224, training: bool = True):
        super().__init__(apply_to=[], training=training)
        self.image_size = image_size

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a LeRobot sample into QwenFast input fields."""
        result = {}

        # 1. observation.images.* → image (List[PIL])
        image_keys = sorted(k for k in data if k.startswith("observation.images."))
        images = []
        for k in image_keys:
            img_tensor = data[k]
            if isinstance(img_tensor, torch.Tensor):
                arr = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                images.append(Image.fromarray(arr))
            elif isinstance(img_tensor, Image.Image):
                images.append(img_tensor)
            else:
                images.append(Image.fromarray(np.array(img_tensor, dtype=np.uint8)))
        result["image"] = images if images else [Image.new("RGB", (self.image_size, self.image_size))]

        # 2. task → lang
        result["lang"] = data.get("task", "")

        # 3. action → numpy (already normalized by ActionTransform in pipeline)
        action = data.get("action")
        if action is not None:
            if isinstance(action, torch.Tensor):
                result["action"] = action.float().numpy()
            else:
                result["action"] = np.asarray(action, dtype=np.float32)
        else:
            result["action"] = np.zeros((1, 7), dtype=np.float32)

        return result
