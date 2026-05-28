"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

import abc
from collections.abc import Iterable
from dataclasses import dataclass, field, asdict
from typing import Any

import torch


@dataclass
class OptimizerConfig(abc.ABC):
    """Base optimizer config."""
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @abc.abstractmethod
    def build(self, params) -> torch.optim.Optimizer:
        """Build and return the optimizer from params."""
        raise NotImplementedError


@dataclass
class XVLAAdamWConfig(OptimizerConfig):
    """Custom AdamW optimizer for XVLA with differential learning rates.

    The Vision-Language Model (VLM) is trained with 1/10 of the base learning rate
    for stable optimization, while all other components use the full LR.
    """
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    soft_prompt_lr_scale: float = 1.0
    soft_prompt_warmup_lr_scale: float | None = None

    def build(self, params) -> torch.optim.Optimizer:
        """Build AdamW optimizer with differential LR groups for VLM, soft_prompts, and other params."""
        assert isinstance(params, dict), "XVLAAdamWConfig requires `named_parameters()` as inputs."
        vlm_group, soft_prompt_group, other_group = [], [], []
        for name, p in params.items():
            if not p.requires_grad:
                continue
            if "vlm" in name.lower():
                vlm_group.append(p)
            elif "soft_prompt" in name.lower():
                soft_prompt_group.append(p)
            else:
                other_group.append(p)

        soft_prompt_lr = self.lr * self.soft_prompt_lr_scale
        if self.soft_prompt_warmup_lr_scale is not None:
            soft_prompt_lr = self.lr * self.soft_prompt_warmup_lr_scale

        param_groups: list[dict[str, Any]] = [
            {"params": vlm_group, "lr": self.lr * 0.1, "weight_decay": self.weight_decay * 0.1, "name": "vlm"},
            {
                "params": soft_prompt_group,
                "lr": soft_prompt_lr,
                "weight_decay": self.weight_decay,
                "name": "soft_prompts",
            },
            {"params": other_group, "lr": self.lr, "weight_decay": self.weight_decay, "name": "other"},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        return torch.optim.AdamW(param_groups, betas=self.betas, eps=self.eps)