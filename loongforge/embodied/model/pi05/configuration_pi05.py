# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from lerobot (https://github.com/huggingface/lerobot).
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Pi05 configuration with task-level parameters and training switches."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Pi05Config:
    """Pi05 training configuration.

    Backbone architecture parameters, such as gemma_2b, gemma_300m expert,
    and image size 224, are fixed in modeling_pi05.py. This config keeps only
    task-level variable fields.

    YAML example (configs/models/embodied/pi05.yaml):
        model_type: pi05
        tokenizer_name: /path/to/paligemma-tokenizer
        action_dim: 7
        state_dim: 7
        action_horizon: 50
    """

    # Paths
    tokenizer_name: str = ""                   # Local PaliGemma tokenizer path

    # Task dimensions
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 50

    # Internal PI05Pytorch padding dimensions, usually matching LeRobot defaults
    max_action_dim: int = 32
    max_state_dim: int = 32

    # Training switches
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
