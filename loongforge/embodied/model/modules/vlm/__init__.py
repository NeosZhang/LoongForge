# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
VLM (Vision-Language Model) module for LoongForge.

Provides a unified interface for the supported VLM backend:
    - Qwen2.5-VL (see qwen2_5_vl.py for the implementation)

The model id (HuggingFace repo or local path) is read from
``config.backbone.base_vlm`` inside the VLM interface itself, where it
is used to load the pretrained weights via ``from_pretrained(...)``.
The value is expected to be injected from the launch shell via dotlist
override (e.g., ``backbone.base_vlm=$BASE_VLM_PATH`` in
``examples/embodied/fast/run_fast_ddp.sh``); the YAML config does not
need to define this field.
"""


def get_vlm_model(config):
    """
    Factory: return the VLM interface for the current model.

    Args:
        config: Framework configuration. The VLM interface itself
                reads ``config.backbone.base_vlm`` to determine the
                model id used for weight loading.

    Returns:
        VLM interface instance (currently always :class:`Qwen25VLInterface`).
    """
    # TODO: add Qwen3-VL support (qwen3_vl.py) and dispatch by VLM family.
    from .qwen2_5_vl import Qwen25VLInterface
    return Qwen25VLInterface(config)
