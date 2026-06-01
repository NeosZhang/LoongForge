# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied Model - Unified model building entry point.
"""

import torch.nn as nn


def build_model(cfg) -> nn.Module:
    """Unified model build entry point.
    """
    arch_name = cfg.get("architecture", None)
    # Compose mode: builder pattern with condition/action layering
    from loongforge.embodied.model.compose.builder import build_framework
    return build_framework(cfg)

