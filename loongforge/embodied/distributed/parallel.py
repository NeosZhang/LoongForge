# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DDP/FSDP wrapping with mixed precision managed by the parallel strategy."""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .context import DistributedContext


def wrap_model(model: nn.Module, args, ctx: DistributedContext) -> nn.Module:
    """Wrap model with DDP or FSDP based on CLI args; mixed precision included."""
    dtype = _resolve_dtype(args.dtype)

    if not ctx.is_distributed:
        return model.to(dtype=dtype, device=ctx.device)

    strategy = args.distributed_strategy
    if strategy == "fsdp":
        return _wrap_fsdp(model, args, ctx, dtype)
    else:
        return _wrap_ddp(model, args, ctx, dtype)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP/FSDP wrapper to get raw model."""
    if hasattr(model, "module"):
        return model.module
    return model


def _wrap_ddp(model: nn.Module, args, ctx: DistributedContext, dtype: torch.dtype) -> nn.Module:
    """Wrap model with DistributedDataParallel."""
    model = model.to(dtype=dtype, device=ctx.device)
    return DDP(model, device_ids=[ctx.local_rank], find_unused_parameters=True)


def _wrap_fsdp(model: nn.Module, args, ctx, dtype: torch.dtype) -> nn.Module:
    """
    Apply FSDP2 (fully_shard) to the model.

    Only shards at root level because Pi05 compute_layer_complete() accesses
    layer sub-components directly without calling layer(), preventing FSDP2
    pre-forward hooks from triggering.
    """

    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh

    # Use fp32 master weights
    model.to(dtype=torch.float32, device=ctx.device)

    # Patch modules incompatible with DTensor (e.g. PiGemmaRMSNorm)
    _patch_dtensor_incompatible_modules(model)

    dp_mesh = init_device_mesh(
        "cuda",
        (ctx.world_size,),
        mesh_dim_names=("dp",),
    )

    # Mixed Precision Policies
    mp_default = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
    )

    mp_fp32 = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )

    reshard = _resolve_reshard(args)

    fsdp_kwargs = dict(
        mesh=dp_mesh,
        reshard_after_forward=reshard,
    )

    # Only shard the minimal safe unit (avoid layer-level sharding)
    wrapped_modules = set()

    def _safe_fully_shard(module, mp_policy):
        """Shard module only once, skip if already wrapped."""
        if module in wrapped_modules:
            return
        fully_shard(module, mp_policy=mp_policy, **fsdp_kwargs)
        wrapped_modules.add(module)

    # fp32 top-level subtrees (vision_tower, multi_modal_projector)
    fp32_names = _get_fp32_top_modules(model)

    for name in fp32_names:
        submodule = _get_submodule(model, name)
        if submodule is not None:
            _safe_fully_shard(submodule, mp_fp32)

    # Root wrap
    _safe_fully_shard(model, mp_default)

    return model


def _get_fp32_top_modules(model: nn.Module) -> list:
    """Get names of top-level submodules that should use fp32 compute.

    Identifies modules whose ALL parameters are fp32 and that are direct
    children (not nested inside ModuleLists which get layer-level sharding).
    """
    fp32_names = []
    sharded_ids = set()

    for name, module in model.named_modules():
        if module is model or id(module) in sharded_ids:
            continue
        # Skip modules inside ModuleLists (they will be sharded at layer level)
        if _is_inside_modulelist(model, name):
            continue
        params = list(module.parameters())
        if not params:
            continue
        if all(p.dtype == torch.float32 for p in params):
            fp32_names.append(name)
            for _, child in module.named_modules():
                sharded_ids.add(id(child))

    return fp32_names


def _is_inside_modulelist(model: nn.Module, target_name: str) -> bool:
    """Check if a named module is nested inside any ModuleList."""
    parts = target_name.split(".")
    current = model
    for part in parts[:-1]:
        current = getattr(current, part)
        if isinstance(current, nn.ModuleList):
            return True
    return False


def _get_submodule(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by dot-separated name."""
    parts = name.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current


def _resolve_reshard(args) -> bool:
    """Map fsdp_sharding arg to reshard_after_forward behavior."""
    sharding = getattr(args, "fsdp_sharding", "FULL_SHARD")
    # FULL_SHARD: reshard params after forward (saves memory)
    # SHARD_GRAD_OP: keep params unsharded after forward (faster, more memory)
    return sharding != "SHARD_GRAD_OP"


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def _patch_dtensor_incompatible_modules(model: nn.Module):
    """Patch modules that access params in ways incompatible with DTensor.

    FSDP2 converts params to DTensors. Modules like PiGemmaRMSNorm do
    normed * (1.0 + self.weight.float()) which mixes DTensor with regular Tensor.
    We patch their forward to handle DTensor params before use.
    """
    for module in model.modules():
        if module.__class__.__name__ == "PiGemmaRMSNorm" and hasattr(module, "weight"):
            _patch_rmsnorm_forward(module)


def _patch_rmsnorm_forward(module: nn.Module):
    """Replace PiGemmaRMSNorm.forward with a DTensor-safe version."""
    from torch.distributed.tensor import DTensor

    eps = module.eps

    def _safe_forward(x, cond=None):
        """DTensor-safe RMSNorm forward."""
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + eps)

        if cond is None or module.dense is None:
            w = module.weight
            w_local = w.full_tensor() if isinstance(w, DTensor) else w
            normed = normed * (1.0 + w_local.float())
            return normed.type_as(x), None

        modulation = module.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(x.dtype), gate.to(x.dtype)

    module.forward = _safe_forward
