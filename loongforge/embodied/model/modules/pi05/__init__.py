# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PI05 model package migrated from PiVLA.

Public API:
    PI05Config              - dataclass of all hyperparameters
    PI05Pytorch             - core PyTorch model (training + inference)
    PaliGemmaWithExpertModel - joint-attention PaliGemma + Gemma expert
    prepare_state_prompt / prepare_batch_state_prompts - prompt construction
    tokenize_prompts        - HuggingFace tokenizer wrapper
    build_tokenizer         - build tokenizer from PI05Config
"""

from .configuration_pi05 import PI05Config, DEFAULT_IMAGE_SIZE
from .modeling_pi05 import (
    PI05Pytorch,
    PaliGemmaWithExpertModel,
    GemmaConfig,
    get_gemma_config,
    pad_vector,
    create_sinusoidal_pos_embedding,
    sample_beta,
    make_att_2d_masks,
    resize_with_pad_torch,
    OPENPI_ATTENTION_MASK_VALUE,
)
from .processor_pi05 import (
    prepare_state_prompt,
    prepare_batch_state_prompts,
    tokenize_prompts,
    build_tokenizer,
    discretize_state,
)

__all__ = [
    "PI05Config",
    "DEFAULT_IMAGE_SIZE",
    "PI05Pytorch",
    "PaliGemmaWithExpertModel",
    "GemmaConfig",
    "get_gemma_config",
    "pad_vector",
    "create_sinusoidal_pos_embedding",
    "sample_beta",
    "make_att_2d_masks",
    "resize_with_pad_torch",
    "OPENPI_ATTENTION_MASK_VALUE",
    "prepare_state_prompt",
    "prepare_batch_state_prompts",
    "tokenize_prompts",
    "build_tokenizer",
    "discretize_state",
]
