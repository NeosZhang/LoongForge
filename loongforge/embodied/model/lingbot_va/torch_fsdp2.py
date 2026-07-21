# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""Final block+root native FSDP2 wrapping for LingBot-VA."""

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from loongforge.embodied.distributed.utils import module_params
from loongforge.embodied.model.lingbot_va.wan_model import WanTransformerBlock


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def _rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _ensure_fsdp_param_compat():
    """Patch FSDP2 grad accumulation for PyTorch builds with stale private access."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
        return
    if getattr(FSDPParam, "_lingbot_accum_grad_compat", False):
        return

    def _to_accumulated_grad_if_needed(self):
        unsharded_param = getattr(self, "_unsharded_param", None)
        if (
            self.reduce_dtype is None
            or unsharded_param is None
            or unsharded_param.grad is None
            or unsharded_param.grad.dtype == self.reduce_dtype
        ):
            return
        unsharded_grad = unsharded_param.grad
        unsharded_param.grad = None
        self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    FSDPParam.to_accumulated_grad_if_needed = _to_accumulated_grad_if_needed
    FSDPParam._lingbot_accum_grad_compat = True


def _build_embodied_device_mesh(training_args, ctx):
    shard_size = getattr(training_args, "hsdp_shard_size", None)
    if shard_size is None:
        return init_device_mesh("cuda", (ctx.world_size,), mesh_dim_names=("dp",))
    if shard_size <= 0:
        raise ValueError(f"HSDP shard size must be positive, got {shard_size}.")
    if ctx.world_size % shard_size != 0:
        raise ValueError(
            "HSDP requires world_size to be divisible by hsdp_shard_size, "
            f"got world_size={ctx.world_size}, hsdp_shard_size={shard_size}."
        )
    return init_device_mesh(
        "cuda",
        (ctx.world_size // shard_size, shard_size),
        mesh_dim_names=("replica", "shard"),
    )


def _save_custom_attrs(module):
    return {name: dict(vars(param)) for name, param in module.named_parameters()}


def _restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        for attr_name, attr_value in custom_attrs.get(name, {}).items():
            setattr(param, attr_name, attr_value)


def wrap_lingbot_torch_nested_fsdp2(model, training_args, ctx):
    """Apply the phase4 block+root FSDP2 order and mixed-precision policy."""
    if getattr(training_args, "distributed_strategy", None) != "fsdp":
        raise RuntimeError(
            "LingBot native nested FSDP2 requires embodied FSDP strategy"
        )

    dtype = _resolve_dtype(training_args.dtype)
    if not getattr(ctx, "is_distributed", False):
        return model.to(device=ctx.device)

    _ensure_fsdp_param_compat()
    model.to(device=ctx.device)
    fsdp_kwargs = {
        "mesh": _build_embodied_device_mesh(training_args, ctx),
        "reshard_after_forward": False,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=dtype,
            cast_forward_inputs=False,
        ),
    }

    attrs = _save_custom_attrs(model)
    wrapped_params = set()
    wrapped_param_ids = set()

    def nested_fully_shard(module):
        shard_kwargs = dict(fsdp_kwargs)
        params_before = list(
            module_params(module, excluded_param_ids=wrapped_param_ids)
        )
        if wrapped_params:
            shard_kwargs["ignored_params"] = wrapped_params
        fully_shard(module, **shard_kwargs)
        for param in params_before:
            wrapped_params.add(param)
            wrapped_param_ids.add(id(param))

    wrapped_blocks = 0
    for sub_module in model.modules():
        if isinstance(sub_module, WanTransformerBlock):
            nested_fully_shard(sub_module)
            wrapped_blocks += 1
    nested_fully_shard(model)
    _restore_custom_attrs(model, attrs)

    if _rank0():
        print(
            "LingBot native torch nested FSDP2 wrap enabled "
            f"blocks={wrapped_blocks} child_wrap=none "
            "reshard_after_forward=False keep_fp32_params=True "
            f"mp_policy_param_dtype={dtype} mp_policy_reduce_dtype={dtype} ignored_params=True.",
            flush=True,
        )
    return model
