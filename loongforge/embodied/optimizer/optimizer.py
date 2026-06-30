# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer construction."""

import logging

import torch
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from loongforge.embodied.optimizer.lr import build_param_groups

logger = logging.getLogger(__name__)

OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


class _MultiDtypeZeroOptimizer(torch.optim.Optimizer):
    """Wraps multiple ZeroRedundancyOptimizers (one per dtype) as a single Optimizer."""

    _is_multi_dtype_zero_optimizer = True

    def __init__(self, optimizers):
        """Compose multiple ZeRO optimizers without a flat params list.

        torch.optim.Optimizer.__init__ requires a flat params iterable,
        which we cannot provide here (params are split by dtype across child
        optimizers). Initialise the required attributes manually instead.
        """
        self._optimizers = optimizers
        # Attributes expected by torch.optim.Optimizer and LR schedulers.
        self._refresh_public_optimizer_state()
        self.state = {}
        self._hook_for_profile = None  # expected by some PyTorch internals

    def _refresh_public_optimizer_state(self):
        """Refresh Optimizer-like public attributes from child optimizers."""
        self.param_groups = []
        for opt in self._optimizers:
            self.param_groups.extend(opt.param_groups)
        # ``defaults`` is read by some schedulers, e.g. Transformers'
        # cosine_with_min_lr computes ``min_lr / optimizer.defaults["lr"]``.
        # The scheduler still updates every entry in the merged ``param_groups``
        # above, so inheriting defaults from the first child optimizer does not
        # limit LR decay to the first child optimizer.
        #
        # TODO: Support explicit per-param-group LR scheduler settings so
        # min-lr floors can be configured per group instead of deriving one
        # global ratio from ``defaults["lr"]``.
        self.defaults = dict(self._optimizers[0].defaults) if self._optimizers else {}

    def zero_grad(self, set_to_none=True):
        """Clear gradients of all child optimizers."""
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError(
                "_MultiDtypeZeroOptimizer does not support closure-based optimizers."
            )
        for opt in self._optimizers:
            opt.step()
        return None

    def state_dict(self):
        """Return a list of state_dicts, one per child optimizer (order matches __init__)."""
        return [opt.state_dict() for opt in self._optimizers]

    def load_state_dict(self, state_dict):
        """Restore each child optimizer from the corresponding state_dict, then refresh param_groups."""
        if len(state_dict) != len(self._optimizers):
            raise ValueError(
                f"Expected {len(self._optimizers)} optimizer state dicts, got {len(state_dict)}."
            )
        for opt, opt_state_dict in zip(self._optimizers, state_dict):
            opt.load_state_dict(opt_state_dict)
        self._refresh_public_optimizer_state()

    def consolidate_state_dict(self, to=0):
        """Consolidate each child ZeRO optimizer state dict."""
        for opt in self._optimizers:
            opt.consolidate_state_dict(to=to)


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Build the training optimizer.

    - Build per-module LR parameter groups via ``build_param_groups``.
    - Select AdamW/Adam/SGD from ``args.optimizer``.
    - Use ZeRO-1 to shard optimizer states when ``--zero-optimizer`` is enabled with DDP.
    - Split mixed-dtype parameters into per-dtype ZeRO optimizers to satisfy ZeroRedundancyOptimizer dtype constraints.
    """

    groups = build_param_groups(model, args)
    optimizer_cls = OPTIMIZER_REGISTRY.get(args.optimizer)
    if optimizer_cls is None:
        supported = ", ".join(OPTIMIZER_REGISTRY)
        raise ValueError(f"Unknown optimizer '{args.optimizer}'. Supported optimizers: {supported}.")

    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if optimizer_cls in (torch.optim.AdamW, torch.optim.Adam):
        kwargs.update(betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)

    dtype_stats = {}
    for group in groups:
        for p in group.get("params", []):
            if p.requires_grad:
                stats = dtype_stats.setdefault(p.dtype, {"tensors": 0, "elements": 0})
                stats["tensors"] += 1
                stats["elements"] += p.numel()
    param_dtypes = set(dtype_stats)
    if _is_rank_zero():
        summary = ", ".join(
            f"{dtype}: {stats['tensors']} tensors/{stats['elements']} elems"
            for dtype, stats in sorted(dtype_stats.items(), key=lambda item: str(item[0]))
        )
        logger.info("Optimizer trainable parameter dtypes: %s", summary or "none")

    # ZeRO Stage-1: shard optimizer states across DDP ranks
    use_zero = getattr(args, "zero_optimizer", False)
    if use_zero:
        strategy = getattr(args, "distributed_strategy", "ddp")
        parameters_as_bucket_view = getattr(args, "zero_parameters_as_bucket_view", False)
        if strategy == "ddp":
            # Check for mixed dtype parameters - ZeroRedundancyOptimizer requires uniform dtype
            if len(param_dtypes) > 1:
                # Mixed dtype: split param groups by dtype, one ZeRO optimizer per dtype
                # (mirrors the pattern in mixed_precision_train.py)
                logger.info(
                    f"Mixed dtype params {param_dtypes}: using per-dtype ZeroRedundancyOptimizer"
                )
                dtype_groups = _split_param_groups_by_dtype(groups)
                opts = [
                    ZeroRedundancyOptimizer(
                        dtype_group,
                        optimizer_class=optimizer_cls,
                        parameters_as_bucket_view=parameters_as_bucket_view,
                        **kwargs,
                    )
                    for dtype_group in dtype_groups.values()
                ]
                return _MultiDtypeZeroOptimizer(opts)
            else:
                logger.info("Using ZeroRedundancyOptimizer (ZeRO Stage-1) with DDP")
                return ZeroRedundancyOptimizer(
                    groups,
                    optimizer_class=optimizer_cls,
                    parameters_as_bucket_view=parameters_as_bucket_view,
                    **kwargs,
                )
        else:
            logger.warning(
                f"--zero-optimizer ignored: only effective with --distributed-strategy ddp, "
                f"current strategy is '{strategy}' (already shards optimizer states)."
            )

    return optimizer_cls(groups, **kwargs)


def _is_rank_zero() -> bool:
    """Return True on rank 0 without requiring distributed initialization."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _split_param_groups_by_dtype(groups: list[dict]) -> dict[torch.dtype, list[dict]]:
    """Split optimizer param groups by dtype while preserving per-group options.

    Only trainable parameters are split; non-trainable ones are skipped
    (they will not be updated anyway and must not be passed to ZeRO).
    """
    dtype_groups: dict[torch.dtype, list[dict]] = {}
    for group in groups:
        base_group = {k: v for k, v in group.items() if k != "params"}
        params_by_dtype: dict[torch.dtype, list[nn.Parameter]] = {}
        for p in group.get("params", []):
            if not p.requires_grad:
                continue
            params_by_dtype.setdefault(p.dtype, []).append(p)
        for dtype, params in params_by_dtype.items():
            dtype_groups.setdefault(dtype, []).append({**base_group, "params": params})
    return dtype_groups
