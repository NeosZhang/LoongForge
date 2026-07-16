# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 finetune trainer.

Extends :class:`FinetuneTrainer` with GR00T-N1.7-specific overrides:

1. ``_wrap_model_for_training``: preserves original parameter dtypes of
   ``GrootN1d7Policy`` by moving to device without a dtype cast, avoiding
   silent fp32→bf16 downcast of trainable parameters.

2. ``_build_optimizer``: uses HuggingFace-style AdamW decay grouping (bias and
   norm parameters excluded from weight decay), mirroring the Isaac-GR00T
   baseline optimizer configuration.

All other training infrastructure (data, scheduler, checkpointing) is
inherited unchanged from :class:`FinetuneTrainer`.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from loongforge.embodied.distributed.utils import filter_supported_kwargs, unwrap_model
from loongforge.embodied.distributed.utils import parse_optional_int_list
from loongforge.embodied.optimizer.lr_scheduler import build_param_groups
from loongforge.embodied.optimizer.optimizer import OPTIMIZER_REGISTRY
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer

logger = logging.getLogger(__name__)

# Parameter name patterns that should NOT receive weight decay (bias and norms).
_NO_DECAY_NAME_PATTERNS = (
    "bias",
    "layernorm",
    "rmsnorm",
    r"(?:^|\.)norm(?:$|\.)",
    r"_norm(?:$|\.)",
)


def _split_hf_decay_groups(
    model: nn.Module,
    params: List[nn.Parameter],
    lr: float,
    weight_decay: float,
) -> List[Dict]:
    """Split params into HuggingFace AdamW decay / no-decay groups.

    Bias and norm-related parameters (LayerNorm, RMSNorm, etc.) go into
    the no-decay group with ``weight_decay=0.0``; all other parameters go
    into the decay group with the specified ``weight_decay``.
    """
    param_ids = {id(p) for p in params}

    # Build the set of parameter names that should receive weight decay.
    no_decay_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) not in param_ids:
            continue
        # LayerNorm instances never decay.
        if isinstance(parameter, nn.LayerNorm):
            no_decay_ids.add(id(parameter))
            continue
        # Check name-based patterns.
        for pattern in _NO_DECAY_NAME_PATTERNS:
            if re.search(pattern, name):
                no_decay_ids.add(id(parameter))
                break

    decay_params = [p for p in params if id(p) not in no_decay_ids]
    no_decay_params = [p for p in params if id(p) in no_decay_ids]

    groups: List[Dict] = []
    if decay_params:
        groups.append({
            "params": decay_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": "hf_decay",
        })
    if no_decay_params:
        groups.append({
            "params": no_decay_params,
            "lr": lr,
            "weight_decay": 0.0,
            "name": "hf_no_decay",
        })
    return groups


class GrootN1d7Trainer(FinetuneTrainer):
    """GR00T-N1.7 finetune trainer.

    Overrides:
    - ``_wrap_model_for_training``: dtype-preserving DDP wrap.
    - ``_build_optimizer``: HuggingFace-style AdamW decay grouping.
    - ``_init_data_iterator`` / ``_advance_epoch``: reset DataLoader worker
      RNG via ``model.reset_data_iterator_rng`` to align with Isaac baseline.
    """

    def _init_data_iterator(self, name: str):
        """Reset DataLoader RNG before initializing iterator."""
        self._maybe_reset_data_iterator_rng()
        super()._init_data_iterator(name)

    def _advance_epoch(self, name: str):
        """Reset DataLoader RNG before advancing to the next epoch."""
        self._maybe_reset_data_iterator_rng()
        super()._advance_epoch(name)

    def _maybe_reset_data_iterator_rng(self) -> None:
        """Reset DataLoader worker base seeds via model.reset_data_iterator_rng.

        ``GrootN1d7Policy.reset_data_iterator_rng`` resets Python/NumPy/Torch
        RNG to align worker seed initialization with the Isaac-GR00T baseline.
        This is a no-op for models that do not implement the method.
        """
        policy = unwrap_model(self.model)
        reset_fn = policy.reset_data_iterator_rng
        if reset_fn is not None:
            reset_fn(self.training_args.seed)

    def _wrap_model_for_training(self) -> None:
        """Move model to device without dtype cast, then wrap with DDP/no-op.

        Skips the ``wrap_model`` call from ``parallel.py`` to avoid the
        ``model.to(dtype=dtype)`` path that would overwrite fp32 trainable
        parameters. Instead:

        1. Move model to device only (``.to(device=...)``, no dtype change).
        2. In distributed mode, broadcast parameters from rank 0 to align
           all ranks, then wrap with DDP using the same kwargs as the
           standard path.
        3. In single-process mode, no wrapping is needed.
        """
        ctx = self.ctx
        training_args = self.training_args

        # Move to device without touching dtype.
        self.model = self.model.to(device=ctx.device)

        if not ctx.is_distributed:
            return

        # Broadcast parameters from rank 0 so all ranks start with identical
        # weights before DDP wraps and hooks are installed.
        with torch.no_grad():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)

        ddp_kwargs = {
            "broadcast_buffers": training_args.ddp_broadcast_buffers,
            "init_sync": training_args.ddp_init_sync,
            "bucket_cap_mb": training_args.ddp_bucket_cap_mb,
            "find_unused_parameters": training_args.ddp_find_unused_parameters,
            "gradient_as_bucket_view": training_args.ddp_gradient_as_bucket_view,
            "static_graph": training_args.ddp_static_graph,
            "skip_all_reduce_unused_params": training_args.ddp_skip_all_reduce_unused_params,
            "bucket_cap_mb_list": parse_optional_int_list(training_args.ddp_bucket_cap_mb_list),
            "batched_grad_copy": training_args.ddp_batched_grad_copy,
        }

        self.model = DDP(self.model, **filter_supported_kwargs(DDP, ddp_kwargs))
        logger.info(
            "GrootN1d7Trainer: model wrapped with DDP (dtype-preserving path, "
            "no dtype cast applied)."
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with HuggingFace-style decay grouping.

        Bias and norm parameters (LayerNorm, RMSNorm, ``*norm*``) are placed
        in a no-decay group with ``weight_decay=0.0``; all other trainable
        parameters receive ``training_args.weight_decay``. This mirrors the
        Isaac-GR00T baseline optimizer configuration.

        Per-module LR groups from ``--lr-group`` are applied first (via the
        public ``build_param_groups``); only the residual "base" group is
        split into decay / no-decay here.
        """
        training_args = self.training_args
        raw_model = unwrap_model(self.model)

        # build_param_groups handles --lr-group entries. When adamw_decay_style
        # is "all" (the public default), it returns a single "base" group for
        # remaining params. We post-process that group into hf_decay/hf_no_decay.
        groups = build_param_groups(raw_model, training_args)

        # Re-split the "base" group (and any group without explicit weight_decay)
        # into hf_decay / hf_no_decay. --lr-group groups keep their own lr but
        # also get the decay split applied.
        final_groups: List[Dict] = []
        for group in groups:
            if "weight_decay" in group:
                # Already has explicit weight_decay (e.g. set by --lr-group
                # or a previous split); keep as-is.
                final_groups.append(group)
            else:
                lr = group.get("lr", training_args.lr_base)
                final_groups.extend(
                    _split_hf_decay_groups(
                        raw_model,
                        group["params"],
                        lr,
                        training_args.weight_decay,
                    )
                )

        optimizer_name = training_args.optimizer
        if optimizer_name not in OPTIMIZER_REGISTRY:
            supported = ", ".join(OPTIMIZER_REGISTRY)
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. Supported: {supported}."
            )
        optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]
        if optimizer_cls is None:
            raise ImportError(
                f"Optimizer '{optimizer_name}' is not available: "
                "the corresponding backend is not installed."
            )

        # weight_decay is already encoded per-group; pass only lr + betas/eps.
        kwargs: Dict = {"lr": training_args.lr_base}
        if optimizer_cls in (torch.optim.AdamW, torch.optim.Adam):
            kwargs.update(
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_eps,
            )
            if optimizer_name == "TorchFusedAdamW":
                kwargs["fused"] = True
        elif optimizer_name in ("TEFusedAdamW", "ApexFusedAdamW"):
            kwargs.update(
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_eps,
            )
            kwargs["adam_w_mode"] = True

        optimizer = optimizer_cls(final_groups, **kwargs)
        logger.info(
            "GrootN1d7Trainer: built %s with hf-style decay grouping "
            "(%d param groups).",
            optimizer_name,
            len(final_groups),
        )
        return optimizer
