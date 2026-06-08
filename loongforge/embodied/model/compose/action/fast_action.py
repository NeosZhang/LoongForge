# Adaptted from 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Fast Action Tokenizer Adapter
"this file is adapted from https://huggingface.co/physical-intelligence/fast"

Overview:
    This module encapsulates a lightweight "action -> language model-readable sequence" converter
    (FastActionTokenizer). Its core objective is to convert continuous/discrete raw robot actions
    (raw_actions) into pseudo-natural language token strings like
    <robot_action_12><robot_action_3><robot_action_87> ... This facilitates direct integration
    into multimodal large models (VLM/LLM) dialogue templates, leveraging their language modeling
    capabilities for action prediction.
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, PreTrainedTokenizerFast

from loongforge.embodied.model.compose.action.base import BaseAction
from loongforge.embodied.train.global_vars import get_args


# Action token range in the VLM vocab (FAST action tokens occupy 151665..153712).
# Canonical source of truth: see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
_ACTION_TOKEN_MIN = 151665
_ACTION_TOKEN_MAX = 153712


def _load_fast_processor(pretrained_path: str = "physical-intelligence/fast"):
    """Load the FAST UniversalActionProcessor with compatibility for transformers >= 5.x.

    transformers 5.x changed AutoProcessor internals which breaks the default
    loading path for the physical-intelligence/fast custom processor. This
    helper manually loads the custom class and its BPE tokenizer component.
    """
    try:
        return AutoProcessor.from_pretrained(pretrained_path, trust_remote_code=True)
    except (ValueError, OSError):
        pass

    # Fallback: manual load
    from huggingface_hub import snapshot_download
    import importlib.util

    local_dir = snapshot_download(pretrained_path)

    spec = importlib.util.spec_from_file_location(
        "processing_action_tokenizer",
        os.path.join(local_dir, "processing_action_tokenizer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    UniversalActionProcessor = mod.UniversalActionProcessor

    bpe_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(local_dir, "tokenizer.json"),
        clean_up_tokenization_spaces=False,
    )

    with open(os.path.join(local_dir, "processor_config.json"), "r") as f:
        cfg = json.load(f)

    processor = UniversalActionProcessor(
        bpe_tokenizer=bpe_tokenizer,
        scale=cfg.get("scale", 10),
        vocab_size=cfg.get("vocab_size", 2048),
        min_token=cfg.get("min_token", -354),
        action_dim=cfg.get("action_dim"),
        time_horizon=cfg.get("time_horizon"),
    )
    return processor


class FastActionTokenizer(BaseAction):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, fast_tokenizer_name="playground/Pretrained_models/fast"):
        # BaseAction.__init__ requires (config, action_head). FastActionTokenizer
        # is an action tokenizer module (Layer 1), not an action strategy
        # (Layer 4); it has no strategy-level config or action_head. Pass None
        # to satisfy the base contract without introducing fake state.
        super().__init__(config=None, action_head=None)

        self.fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_name, trust_remote_code=True
        )  # load https://huggingface.co/physical-intelligence/fast

        # Action token range (used by decoder_action)
        self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
        self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def compute_loss(
        self,
        action_context: torch.Tensor,
        target_actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Placeholder: required by BaseAction ABC contract.

        FastActionTokenizer is an action tokenizer module (Layer 1), not an
        action strategy (Layer 4). QwenFast is end-to-end — loss is computed
        inside VLM forward(), not via an action strategy. This method is
        never called. Implemented only to satisfy BaseAction's
        @abstractmethod; raises if someone incorrectly wires it into a
        strategy-based trainer.
        """
        raise NotImplementedError(
            "FastActionTokenizer is a tokenizer module, not an action strategy. "
            "QwenFast computes loss end-to-end via VLM forward(); this method "
            "should never be called."
        )

    def predict(
        self,
        action_context: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Placeholder: required by BaseAction ABC contract.

        FastActionTokenizer is an action tokenizer module (Layer 1), not an
        action strategy (Layer 4). QwenFast is end-to-end — inference is done
        via VLM generate(), not via an action strategy. This method is never
        called. Implemented only to satisfy BaseAction's @abstractmethod;
        raises if someone incorrectly wires it into a strategy-based trainer.
        """
        raise NotImplementedError(
            "FastActionTokenizer is a tokenizer module, not an action strategy. "
            "QwenFast generates actions end-to-end via VLM generate(); this "
            "method should never be called."
        )

    def encoder_action2fastoken(self, raw_actions):
        """Encode raw actions into FAST token strings.

        Args:
            raw_actions: Iterable of arrays with shape (chunk, dim) per sample.

        Returns:
            list[str]: FAST tokenized action sequences.
        """
        # x: (batch_size, chunck, dim)
        batch_actions = np.stack(raw_actions, axis=0)  # (B, T, D)
        batch_fast_tokens = self.fast_tokenizer(batch_actions)

        return batch_fast_tokens  # List[str]

    def decoder_action(self, generated_ids):
        """Decode generated action token ids back to raw action arrays.

        Args:
            generated_ids: Token ids produced by the language model.

        Returns:
            np.ndarray: Decoded actions with shape (batch_size, chunk, dim).
        """
        # api https://huggingface.co/physical-intelligence/fast
        # return: (batch_size, chunck, dim)
        pred_actions = self.fast_tokenizer.decode([generated_ids - self._ACTION_TOKEN_MIN])
        return pred_actions

    def fit_tokenizer_on_datasets(
        self,
        action_dataset,
        datasets_path="<your_local_path>",
    ):
        """Fit (or load) the underlying FAST tokenizer on the provided action dataset.

        If ``datasets_path`` already exists, the tokenizer is loaded from disk;
        otherwise the tokenizer is fit on ``action_dataset`` and saved to that path.

        Args:
            action_dataset: Dataset used to fit the tokenizer when no saved one exists.
            datasets_path: Local path to load from or save the fitted tokenizer to.
        """
        # If datasets_path exists, load directly
        if os.path.exists(datasets_path):

            self.fast_tokenizer = AutoProcessor.from_pretrained(datasets_path, trust_remote_code=True)
            return
        else:
            # If not found, Fit the tokenizer on the new dataset
            new_tokenizer = self.fast_tokenizer.tokenizer.fit(action_dataset)
            self.fast_tokenizer = new_tokenizer

            # Save the new tokenizer, optionally push it to the Hugging Face model hub
            self.fast_tokenizer.save_pretrained(datasets_path)


def get_action_model(config=None):
    """
    Factory: build FastActionTokenizer from LoongForge config.

    Priority: CLI args (--tokenizer-path) > YAML config > default

    Args:
        config: LoongForge config (expects config.action_model namespace).
    Returns:
        FastActionTokenizer: Initialized FAST tokenizer.
    """
    # Priority 1: CLI args --tokenizer-path
    args = get_args()
    fast_tokenizer_name = getattr(args, "tokenizer_path", None)

    # Priority 2: YAML config
    if fast_tokenizer_name is None and config is not None:
        action_cfg = config.get("action_model", {})
        fast_tokenizer_name = action_cfg.get(
            "fast_tokenizer_path",
            action_cfg.get("fast_tokenizer_name", None)
        )

    # Priority 3: Default
    if fast_tokenizer_name is None:
        fast_tokenizer_name = "physical-intelligence/fast"

    action_model = FastActionTokenizer(fast_tokenizer_name=fast_tokenizer_name)

    return action_model


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

