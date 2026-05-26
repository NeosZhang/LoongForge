# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Migrated from PiVLA/model/pi05/processor_pi05.py.
# Simplified: removed lerobot processor pipeline dependency.

"""Standalone pre/post-processing utilities for PI05.

Key entry points:
    prepare_state_prompt(state, task_str, max_state_dim) -> str
    prepare_batch_state_prompts(states, tasks, max_state_dim) -> list[str]
    tokenize_prompts(prompts, tokenizer, max_length) -> dict
    build_tokenizer(config) -> AutoTokenizer
    discretize_state(state, num_bins) -> ndarray
"""

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from .configuration_pi05 import PI05Config
from .modeling_pi05 import pad_vector


def discretize_state(state: torch.Tensor, num_bins: int = 256) -> np.ndarray:
    """Discretize normalized state in [-1, 1] into integer bin indices."""
    state_np = state.cpu().numpy()
    bins = np.linspace(-1, 1, num_bins + 1)[:-1]
    return np.digitize(state_np, bins=bins) - 1


def prepare_state_prompt(
    state: torch.Tensor,
    task: str,
    max_state_dim: int = 32,
) -> str:
    """Build the full language prompt for PI05 (single sample).

    Format: "Task: <task>, State: <d0> <d1> ...;\nAction: "
    """
    state = deepcopy(state)
    state = pad_vector(state, max_state_dim)
    discretized = discretize_state(state)
    cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(map(str, discretized))
    return f"Task: {cleaned_text}, State: {state_str};\nAction: "


def prepare_batch_state_prompts(
    states: torch.Tensor,
    tasks: list,
    max_state_dim: int = 32,
) -> list:
    """Build prompts for a batch."""
    prompts = []
    for i, task in enumerate(tasks):
        prompts.append(prepare_state_prompt(states[i], task, max_state_dim))
    return prompts


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
