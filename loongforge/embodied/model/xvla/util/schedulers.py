"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

import abc
import logging
import math
from dataclasses import dataclass

import torch


@dataclass
class LRSchedulerConfig(abc.ABC):
    """Base class for LR scheduler configs."""
    num_warmup_steps: int | None


@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Cosine decay with linear warmup scheduler."""
    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer, num_training_steps: int):
        """Build and return a LambdaLR scheduler with cosine decay and linear warmup."""
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

        def lr_lambda(current_step):
            if current_step < actual_warmup_steps:
                frac = 1 - current_step / max(actual_warmup_steps, 1)
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1
            step = min(current_step, actual_decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
            alpha = self.decay_lr / self.peak_lr
            return (1 - alpha) * cosine_decay + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)