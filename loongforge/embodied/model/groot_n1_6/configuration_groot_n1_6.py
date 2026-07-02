#!/usr/bin/env python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""Configuration classes for GR00T-N1.6 embodied model."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict


_DEFAULT_EAGLE_ASSETS = "aravindhs-NV/eagle3-processor-groot-n1d6"
_DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.6-3B"


@dataclass
class GrootN1d6Config:
    """GR00T-N1.6 training configuration.

    This is the single configuration object consumed by the embodied trainer,
    model, and preprocessor. Fixed Megatron/HF compatibility fields from the
    legacy wrapper are intentionally kept out of this config.
    """

    model_type: str = "Gr00tN1d6"
    base_model_path: str = _DEFAULT_BASE_MODEL
    model_name: str = _DEFAULT_EAGLE_ASSETS
    vlm_tokenizer_path: str | None = _DEFAULT_EAGLE_ASSETS
    backbone_model_type: str = "eagle"

    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 50
    max_action_dim: int = 128
    max_state_dim: int = 128

    preprocess_action_horizon: int = 16
    preprocess_max_action_dim: int = 29
    preprocess_max_state_dim: int = 29
    groot_preprocess_mode: str = "sample"
    max_token_len: int | None = None

    use_image_transform: bool = True
    image_resize_strategy: str = "none"
    image_normalize_mode: str = "identity"
    use_action_transform: bool = True
    action_apply_to: list[str] = field(default_factory=lambda: ["action"])
    action_normalization_mode: str = "identity"
    action_use_statistics: bool = False
    action_padding_strategy: str = "none"
    action_transform_horizon: int | None = None
    action_transform_max_action_dim: int | None = None

    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    backbone_embedding_dim: int = 2048
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True
    backbone_trainable_params_fp32: bool = True
    tune_top_llm_layers: int = 4
    tune_llm: bool = False
    tune_visual: bool = False

    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2
    diffusion_model_cfg: Dict[str, Any] = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )
    add_pos_embed: bool = True
    use_vlln: bool = True
    max_seq_len: int = 1024
    max_num_embodiments: int = 32

    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0

    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    use_bf16: bool = True

    embodiment_tag: str = "libero_panda"
    formalize_language: bool = True
    use_albumentations_transforms: bool = True
    use_relative_action: bool = True
    apply_sincos_state_encoding: bool = False
    use_processor_image_size: bool = False

    image_crop_size: list[int] | tuple[int, int] | None = field(default_factory=lambda: [224, 224])
    image_target_size: list[int] | tuple[int, int] | None = field(default_factory=lambda: [224, 224])
    shortest_image_edge: int | None = 256
    crop_fraction: float | None = 0.95
    random_rotation_angle: int | None = None
    color_jitter_params: Dict[str, float] | None = field(
        default_factory=lambda: {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        }
    )

    @classmethod
    def from_config(cls, cfg: Any) -> "GrootN1d6Config":
        """Create from OmegaConf/dict/object without nested backbone merging."""
        if isinstance(cfg, cls):
            return cfg
        if hasattr(cfg, "items"):
            items = dict(cfg.items())
        elif isinstance(cfg, dict):
            items = dict(cfg)
        else:
            items = {
                key: getattr(cfg, key)
                for key in cls.__dataclass_fields__
                if hasattr(cfg, key)
            }
        values = {
            key: value
            for key, value in items.items()
            if key in cls.__dataclass_fields__ and key != "_target_"
        }
        return cls(**values)

    def __post_init__(self) -> None:
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
        if checkpoint_path:
            self.base_model_path = checkpoint_path

        eagle_local = os.environ.get("EAGLE_LOCAL_PATH")
        if eagle_local:
            self.model_name = eagle_local
            self.vlm_tokenizer_path = eagle_local
        else:
            if not self.model_name:
                self.model_name = _DEFAULT_EAGLE_ASSETS
            if self.vlm_tokenizer_path is None:
                self.vlm_tokenizer_path = _DEFAULT_EAGLE_ASSETS

        if self.tune_top_llm_layers < 0:
            raise ValueError(f"tune_top_llm_layers ({self.tune_top_llm_layers}) must be non-negative")
