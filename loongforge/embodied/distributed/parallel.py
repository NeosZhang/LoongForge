# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DDP/FSDP wrapping with mixed precision managed by the parallel strategy."""

import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

from .context import DistributedContext
from .utils import (
    resolve_dtype,
    filter_supported_kwargs,
    get_module_names_by_dtype,
    is_container_module,
    module_depth,
    module_param_dtypes,
    module_param_numel,
    module_params,
    parse_optional_int_list,
)

logger = logging.getLogger(__name__)


def wrap_model(model: nn.Module, args, ctx: DistributedContext) -> nn.Module:
    """Wrap model with DDP or FSDP based on CLI args; mixed precision included."""
    dtype = resolve_dtype(args.dtype)

    if not ctx.is_distributed:
        return model.to(dtype=dtype, device=ctx.device)

    strategy = args.distributed_strategy
    if strategy == "fsdp":
        return _wrap_fsdp(model, args, ctx, dtype)
    else:
        return _wrap_ddp(model, args, ctx, dtype)


def _wrap_ddp(model: nn.Module, args, ctx: DistributedContext, dtype: torch.dtype) -> nn.Module:
    """Wrap model with DistributedDataParallel."""
    all_modules_by_dtype = get_module_names_by_dtype(model, trainable_only=False)
    if len(all_modules_by_dtype) > 1:
        model = model.to(device=ctx.device)
    else:
        model = model.to(dtype=dtype, device=ctx.device)

    ddp_kwargs = {
        "broadcast_buffers": getattr(args, "ddp_broadcast_buffers", True),
        "init_sync": getattr(args, "ddp_init_sync", True),
        "bucket_cap_mb": getattr(args, "ddp_bucket_cap_mb", None),
        "find_unused_parameters": getattr(args, "ddp_find_unused_parameters", True),
        "gradient_as_bucket_view": getattr(args, "ddp_gradient_as_bucket_view", False),
        "static_graph": getattr(args, "ddp_static_graph", False),
    }
    if hasattr(args, "ddp_skip_all_reduce_unused_params"):
        ddp_kwargs["skip_all_reduce_unused_params"] = args.ddp_skip_all_reduce_unused_params
    if hasattr(args, "ddp_bucket_cap_mb_list"):
        ddp_kwargs["bucket_cap_mb_list"] = parse_optional_int_list(args.ddp_bucket_cap_mb_list)
    if hasattr(args, "ddp_batched_grad_copy"):
        ddp_kwargs["batched_grad_copy"] = args.ddp_batched_grad_copy

    return DDP(model, **filter_supported_kwargs(DDP, ddp_kwargs))


def _wrap_fsdp(model: nn.Module, args, ctx, dtype: torch.dtype) -> nn.Module:
    """Apply FSDP2 with dtype-safe, bottom-up wrapping.

    Wrapping has two important constraints:
    1. One FSDP communication group may only contain parameters with the same
       original dtype. Mixed-dtype candidates must be split before calling
       ``fully_shard`` on the candidate itself.
    2. ``fully_shard`` installs all-gather/reshard hooks on the wrapped
       module's ``forward``. Some registered modules are only structural
       containers or helper parameter owners and are never called directly by
       ``model.forward``; those modules are unsafe hook boundaries unless the
       user explicitly identifies them as valid wrap targets.

    The planner wraps from inner modules to outer modules so parent groups can
    ignore parameters that already belong to child groups. The stages are:
    1. Wrap user-specified classes from ``--fsdp-wrap-modules`` first, deepest
       match first, because these are explicit execution boundaries.
    2. Auto-wrap repeated, parameter-heavy non-container modules as likely
       layer/block boundaries.
    3. Wrap remaining large leaf parameter owners by dtype, avoiding tiny leaves
       that would create excessive FSDP groups.
    4. Wrap the root last as the catch-all group for any remaining parameters
       and to expose a top-level FSDPModule to trainer/checkpoint code.

    For every candidate, already wrapped inner parameters are excluded, child
    modules are recursively wrapped or descended into until the candidate's
    remaining parameters are dtype-uniform, then ``fully_shard`` is called with
    those inner parameters as ``ignored_params``. This keeps each parameter in
    exactly one FSDP communication group.
    """
    modules_by_dtype = get_module_names_by_dtype(model, trainable_only=False)
    mixed_original_dtype = len(modules_by_dtype) > 1

    if not mixed_original_dtype:
        model.to(dtype=dtype)

    dp_mesh = _build_fsdp_device_mesh(args, ctx)

    fsdp_kwargs = {
        "mesh": dp_mesh,
    }

    # The planner mutates ``model`` in-place by calling ``fully_shard`` on each
    # selected module. It also tracks already wrapped parameters so parent/root
    # groups do not manage those parameters a second time.
    planner = _FSDPWrapPlanner(
        model=model,
        args=args,
        fsdp_kwargs=fsdp_kwargs,
        dtype=dtype,
        mixed_original_dtype=mixed_original_dtype,
    )

    # Order matters: inner groups must be created before parent/root groups so
    # later outer wraps can ignore parameters already assigned to inner groups.
    explicit_group_count = planner.wrap_user_specified_modules()
    repeated_group_count = planner.wrap_repeated_layer_modules()
    leftover_group_count = planner.wrap_leftover_leaf_modules_by_dtype()
    root_group_created = planner.wrap_root()
    logger.info(
        "FSDP wrapped %d module groups "
        "(explicit=%d, repeated=%d, leftover=%d, root=%s).",
        planner.num_wrapped_groups,
        explicit_group_count,
        repeated_group_count,
        leftover_group_count,
        root_group_created,
    )

    return model


def _build_fsdp_device_mesh(args, ctx: DistributedContext):
    """Build the FSDP/HSDP device mesh used by fully_shard.

    FSDP uses a 1D mesh and shards parameters across all data-parallel ranks.
    HSDP uses a 2D mesh: dim 0 is replicated and dim 1 is sharded. Passing
    ``--hsdp-shard-size`` enables HSDP and sets the sharding dimension size.
    """
    shard_size = getattr(args, "hsdp_shard_size", None)
    if shard_size is None:
        return init_device_mesh(
            "cuda",
            (ctx.world_size,),
            mesh_dim_names=("dp",),
        )

    if shard_size <= 0:
        raise ValueError(f"HSDP shard size must be positive, got {shard_size}.")
    if ctx.world_size % shard_size != 0:
        raise ValueError(
            "HSDP requires world_size to be divisible by hsdp_shard_size, "
            f"got world_size={ctx.world_size}, hsdp_shard_size={shard_size}."
        )

    replica_size = ctx.world_size // shard_size
    if replica_size <= 1:
        logger.warning(
            "--hsdp-shard-size is set with one replica group; this is equivalent "
            "to FSDP over the HSDP shard group."
        )

    if dist.get_rank() == 0:
        logger.info(
            "Using HSDP 2D device mesh: replica=%d, shard=%d.",
            replica_size,
            shard_size,
        )

    return init_device_mesh(
        "cuda",
        (replica_size, shard_size),
        mesh_dim_names=("replica", "shard"),
    )


class _FSDPWrapPlanner:
    """Build and apply dtype-valid FSDP2 groups for a module tree.

    The planner is deliberately generic: model-specific knowledge should enter
    through CLI class-name lists instead of hard-coded Python constants.
    """

    def __init__(
        self,
        model: nn.Module,
        args,
        fsdp_kwargs: dict,
        dtype: torch.dtype,
        mixed_original_dtype: bool,
    ):
        self.model = model
        self.args = args
        self.fsdp_kwargs = fsdp_kwargs
        self.mixed_original_dtype = mixed_original_dtype

        # Mixed precision policy is selected per FSDP group. For a uniform-dtype
        # model, all groups follow the requested training dtype. For a model
        # that already has mixed original dtypes, non-fp32 groups preserve their
        # authored dtype to avoid silently changing precision-sensitive modules.
        self.mp_default = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
        self.mp_preserve = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
        self.mp_fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)

        # Stage 2 and stage 3 use separate thresholds:
        # - repeated_min_num_params controls layer/block auto wrapping;
        # - leftover_min_num_params controls large leaf cleanup wrapping.
        # Keeping the leftover threshold configurable avoids wrapping every tiny
        # Linear/Norm leaf, which can increase communication and hook overhead.
        self.repeated_min_num_params = int(getattr(args, "fsdp_min_num_params", 1_000_000))
        self.leftover_min_num_params = int(
            getattr(args, "fsdp_leftover_min_num_params", self.repeated_min_num_params)
        )

        # reshard_after_forward is chosen per group:
        # module class override > root setting > default non-root setting.
        # This lets callers keep selected hot modules unsharded after forward
        # while still using memory-saving resharding elsewhere.
        self.fsdp_reshard_default = getattr(args, "fsdp_reshard_default", None)
        self.fsdp_reshard_root = getattr(args, "fsdp_reshard_root", False)
        self.fsdp_reshard_module_overrides = (
            getattr(args, "fsdp_reshard_module_overrides", None) or {}
        )

        # Classes listed here are not wrapped as a single boundary. The planner
        # descends into their children instead. This is useful for modules whose
        # parameters are read by custom code outside that module's forward hooks.
        extra_no_wrap = getattr(args, "fsdp_no_wrap_modules", None)
        self.no_wrap_module_classes = {
            name.strip() for name in extra_no_wrap.split(",") if name.strip()
        } if extra_no_wrap else set()

        # Track wrapped parameters separately from wrapped modules. Parent/root
        # FSDP groups pass these as ignored_params so each parameter is managed
        # by exactly one communication group.
        self.wrapped_module_ids = set()
        self.wrapped_param_ids = set()
        self.wrapped_params = set()
        self.wrapped_group_count = 0

    @property
    def num_wrapped_groups(self) -> int:
        """Number of FSDP groups created by fully_shard."""
        return self.wrapped_group_count

    def wrap_user_specified_modules(self) -> int:
        """Wrap explicit class-name FSDP boundaries from --fsdp-wrap-modules.

        This is the highest-priority stage. If the user knows a class is a good
        FSDP boundary, wrap it before heuristic stages see its parameters.
        """
        wrap_modules = getattr(self.args, "fsdp_wrap_modules", None)
        if not wrap_modules:
            return 0
        class_names = {name.strip() for name in wrap_modules.split(",") if name.strip()}
        candidates = [
            (name, module)
            for name, module in self._named_modules()
            if name and not is_container_module(module) and module.__class__.__name__ in class_names
        ]

        # Deepest-first keeps child groups independent. If both a parent and one
        # of its children match the explicit list, the child is sharded first and
        # then excluded from the parent's parameter set.
        candidates.sort(key=lambda item: module_depth(item[0]), reverse=True)
        explicit_group_count = 0
        for _, module in candidates:
            if self._wrap_candidate(module):
                explicit_group_count += 1
        return explicit_group_count

    def wrap_repeated_layer_modules(self) -> int:
        """Wrap repeated modules as likely layer boundaries.

        The heuristic targets module classes that appear multiple times and own
        enough parameters collectively to justify an FSDP group. Small sibling
        layers can be grouped together until the group reaches the configured
        threshold. This matches common model structures such as transformer
        decoder layers or vision encoder blocks.
        """
        # Count duplicate references using remove_duplicate=False. Shared module
        # instances are skipped below because sharding the same object through
        # multiple names would make the group selection ambiguous.
        module_occurrences = {}
        for _, module in self.model.named_modules(remove_duplicate=False):
            module_occurrences[id(module)] = module_occurrences.get(id(module), 0) + 1

        # Collect possible repeated layer boundaries. This pass only filters out
        # modules that are clearly unsafe or out of scope; it does not apply the
        # size threshold yet because adjacent sibling layers may be grouped into
        # one FSDP communication group below.
        #
        # Keep modules that:
        # - are not the root, which ``wrap_root()`` handles last;
        # - are not structural containers or user-declared no-wrap classes;
        # - have children, leaving leaf parameter owners for the leftover stage;
        # - are not shared module instances appearing under multiple names;
        # - still own unwrapped parameters.
        repeated_boundary_candidates = [
            (name, module)
            for name, module in self._named_modules()
            if (
                name
                and id(module) not in self.wrapped_module_ids
                and not is_container_module(module)
                and not self._is_no_wrap_module(module)
                and any(True for _ in module.children())
                and module_occurrences.get(id(module), 0) == 1
                and module_param_numel(module, excluded_param_ids=self.wrapped_param_ids) > 0
            )
        ]

        # A single large helper class is not necessarily a repeated layer. Count
        # candidate classes and keep only classes that appear multiple times,
        # e.g. ``SiglipEncoderLayer: 27`` or ``GemmaMLP: 36`` are layer-like,
        # while one-off classes such as ``PI05Pytorch: 1`` are not.
        class_counts = {}
        for _, module in repeated_boundary_candidates:
            class_name = module.__class__.__name__
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        repeated_layer_candidates = [
            (name, module)
            for name, module in repeated_boundary_candidates
            if class_counts.get(module.__class__.__name__, 0) > 1
        ]

        # If nested candidates both satisfy the repeated-layer heuristic, keep
        # only the outermost candidate for this stage. The dtype split logic can
        # still descend later if the outer candidate cannot be wrapped directly.
        candidate_names = {name for name, _ in repeated_layer_candidates}
        repeated_layer_candidates = [
            (name, module)
            for name, module in repeated_layer_candidates
            if not any(
                name.startswith(parent_name + ".")
                for parent_name in candidate_names
                if parent_name != name
            )
        ]

        # Build contiguous sibling runs before applying the size threshold. FSDP
        # list groups should combine only modules that share the same direct
        # parent and appear next to each other with the same class, e.g.
        # ``layers.0, layers.1, layers.2``.
        candidate_by_name = dict(repeated_layer_candidates)
        repeated_layer_groups = []
        for parent_name, parent_module in self._named_modules():
            # ``sibling_candidate_run`` is one contiguous same-class candidate
            # span under this parent. ``_chunk_repeated_layer_run`` later splits
            # it into one or more threshold-sized repeated layer groups.
            sibling_candidate_run = []
            run_class_name = None
            for child_name, child in parent_module.named_children():
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                child_candidate = candidate_by_name.get(full_name)
                if child_candidate is None:
                    # A non-candidate child breaks the contiguous run.
                    repeated_layer_groups.extend(
                        self._chunk_repeated_layer_run(sibling_candidate_run)
                    )
                    sibling_candidate_run = []
                    run_class_name = None
                    continue

                class_name = child_candidate.__class__.__name__
                if sibling_candidate_run and class_name != run_class_name:
                    # Different repeated classes should form separate groups.
                    repeated_layer_groups.extend(
                        self._chunk_repeated_layer_run(sibling_candidate_run)
                    )
                    sibling_candidate_run = []
                sibling_candidate_run.append(child_candidate)
                run_class_name = class_name

            # Flush the final run for this parent.
            repeated_layer_groups.extend(
                self._chunk_repeated_layer_run(sibling_candidate_run)
            )

        repeated_group_count = 0
        for group in repeated_layer_groups:
            repeated_group_count += self._wrap_repeated_layer_group(group)
        logger.info("FSDP wrapping auto-selected repeated module classes.")
        return repeated_group_count

    def _chunk_repeated_layer_run(self, modules: list[nn.Module]) -> list[list[nn.Module]]:
        """Group adjacent repeated layers until each group reaches the threshold."""
        if not modules:
            return []

        groups = []
        current_group = []
        current_numel = 0
        for module in modules:
            numel = module_param_numel(module, excluded_param_ids=self.wrapped_param_ids)
            if numel <= 0:
                continue
            current_group.append(module)
            current_numel += numel
            if current_numel >= self.repeated_min_num_params:
                groups.append(current_group)
                current_group = []
                current_numel = 0

        if current_group:
            if groups:
                groups[-1].extend(current_group)
            elif current_numel >= self.repeated_min_num_params:
                groups.append(current_group)
        return groups

    def _wrap_repeated_layer_group(self, modules: list[nn.Module]) -> int:
        """Wrap one repeated-layer group and return created FSDP group count.

        The input group has already been selected as adjacent repeated layers.
        This method performs the final safety pass before calling
        ``fully_shard``:
        1. Drop modules already assigned to an inner/earlier FSDP group.
        2. Recursively split mixed-dtype children out of each module.
        3. Scan the remaining modules in order and collect contiguous modules
           with the same remaining dtype.
        4. Wrap each same-dtype run as one FSDP communication group.
        """
        modules = [module for module in modules if id(module) not in self.wrapped_module_ids]
        if not modules:
            return 0

        # Each individual layer must expose at most one remaining dtype before
        # it can participate in a list-based FSDP group.
        for module in modules:
            self._make_candidate_params_uniform(module)

        created_group_count = 0
        same_dtype_group = []
        same_dtype = None
        for module in modules:
            dtypes = module_param_dtypes(module, excluded_param_ids=self.wrapped_param_ids)
            if not dtypes:
                continue
            if len(dtypes) > 1:
                # Flush the current same-dtype run, then fall back to the
                # generic candidate wrapper so it can continue recursive dtype
                # splitting for this difficult module.
                created_group_count += self._wrap_same_dtype_module_group(
                    same_dtype_group, same_dtype
                )
                same_dtype_group = []
                same_dtype = None
                created_group_count += int(self._wrap_candidate(module))
                continue

            dtype = next(iter(dtypes))
            if same_dtype_group and dtype != same_dtype:
                # A dtype change starts a new FSDP communication group.
                created_group_count += self._wrap_same_dtype_module_group(
                    same_dtype_group, same_dtype
                )
                same_dtype_group = []
            same_dtype_group.append(module)
            same_dtype = dtype

        created_group_count += self._wrap_same_dtype_module_group(same_dtype_group, same_dtype)
        return created_group_count

    def _wrap_same_dtype_module_group(
        self,
        modules: list[nn.Module],
        dtype: torch.dtype | None,
    ) -> int:
        if not modules or dtype is None:
            return 0
        target = modules[0] if len(modules) == 1 else modules
        return int(self._safe_fully_shard(target, self._policy_for({dtype})))

    def wrap_leftover_leaf_modules_by_dtype(self) -> int:
        """Wrap remaining leaf parameter owners that exceed the leftover threshold.

        This cleanup stage handles large direct parameter owners missed by the
        explicit and repeated-layer stages. It intentionally ignores modules
        with children because those are better handled as execution boundaries
        or by the recursive dtype splitter.
        """
        leftover_group_count = 0
        for name, module in self._named_modules():
            if not name:
                continue
            if id(module) in self.wrapped_module_ids:
                continue
            if is_container_module(module):
                continue
            if any(True for _ in module.children()):
                continue
            if module_param_numel(module) < self.leftover_min_num_params:
                continue
            if self._wrap_candidate(module):
                leftover_group_count += 1
        return leftover_group_count

    def wrap_root(self) -> bool:
        """Wrap the root module last.

        Root wrapping is a catch-all for residual parameters not assigned to
        inner groups. ``ignored_params`` prevents already wrapped inner params
        from being managed twice.
        """
        return self._wrap_candidate(self.model, force=True)

    def _wrap_candidate(self, module: nn.Module, force: bool = False) -> bool:
        """Wrap module after first making its remaining parameters dtype-uniform.

        FSDP2 requires each flattened group to have a single original parameter
        dtype. If a candidate still contains multiple dtypes after ignoring
        already wrapped inner groups, the planner recursively wraps children
        until the candidate's remaining parameters are dtype-uniform.
        """
        # Containers are traversal structure, not execution boundaries. Shard
        # real modules whose forward hooks can trigger FSDP all-gather.
        if id(module) in self.wrapped_module_ids:
            return False
        if is_container_module(module) and module is not self.model:
            return False

        # A no-wrap class is not a dead end. It means "do not use this module as
        # the FSDP hook boundary"; its children may still be valid boundaries.
        if self._is_no_wrap_module(module):
            return self._wrap_child_boundaries(module)

        self._make_candidate_params_uniform(module)
        dtypes = module_param_dtypes(module, excluded_param_ids=self.wrapped_param_ids)
        if not dtypes:
            if force:
                return self._safe_fully_shard(module, self.mp_preserve)
            return False
        if len(dtypes) > 1:
            raise ValueError(
                f"Unable to derive a uniform-dtype FSDP group for "
                f"{module.__class__.__name__}: {dtypes}."
            )
        return self._safe_fully_shard(module, self._policy_for(dtypes))

    def _named_modules(self) -> list[tuple[str, nn.Module]]:
        return list(self.model.named_modules(remove_duplicate=True))

    def _is_no_wrap_module(self, module: nn.Module) -> bool:
        return module is not self.model and module.__class__.__name__ in self.no_wrap_module_classes

    def _policy_for(self, dtypes: set[torch.dtype]) -> MixedPrecisionPolicy:
        # Uniform-dtype models follow the requested training dtype. Mixed-dtype
        # models preserve non-fp32 groups so model-authored precision choices
        # are not silently overwritten.
        if not self.mixed_original_dtype:
            return self.mp_default
        if dtypes == {torch.float32}:
            return self.mp_fp32
        return self.mp_preserve

    def _reshard_after_forward_for(self, module: nn.Module) -> bool | int | None:
        """Resolve the per-group reshard_after_forward setting."""
        class_name = module.__class__.__name__
        if class_name in self.fsdp_reshard_module_overrides:
            return self.fsdp_reshard_module_overrides[class_name]
        if module is self.model:
            return self.fsdp_reshard_root
        return self.fsdp_reshard_default

    def _make_candidate_params_uniform(self, module: nn.Module) -> None:
        """Recursively isolate minority dtype children before wrapping module.

        The dominant dtype, measured by parameter numel, stays in the current
        candidate. Children that only contain other dtypes are wrapped or
        descended into first. After those child groups are marked ignored, the
        current candidate should expose at most one remaining dtype.
        """
        dtypes = module_param_dtypes(module, excluded_param_ids=self.wrapped_param_ids)
        if len(dtypes) <= 1:
            return

        dtype_numel = {}
        for param in module_params(module, excluded_param_ids=self.wrapped_param_ids):
            dtype_numel[param.dtype] = dtype_numel.get(param.dtype, 0) + param.numel()
        target_dtype = max(dtype_numel, key=dtype_numel.get)

        # First pass: isolate children that do not contain the dominant dtype,
        # and recursively split children that still contain multiple dtypes.
        for child in module.children():
            if id(child) in self.wrapped_module_ids:
                continue
            child_dtypes = module_param_dtypes(child, excluded_param_ids=self.wrapped_param_ids)
            if not child_dtypes:
                continue
            if target_dtype not in child_dtypes:
                self._wrap_valid_boundary_or_children(child)
            elif len(child_dtypes) > 1:
                self._make_candidate_params_uniform(child)

        dtypes = module_param_dtypes(module, excluded_param_ids=self.wrapped_param_ids)
        if len(dtypes) <= 1:
            return

        # Second pass: if mixed dtypes remain, wrap any child whose remaining
        # dtype set differs from the dominant dtype. This handles cases where a
        # child contains the dominant dtype plus another dtype after recursion.
        for child in module.children():
            if id(child) in self.wrapped_module_ids:
                continue
            child_dtypes = module_param_dtypes(child, excluded_param_ids=self.wrapped_param_ids)
            if child_dtypes and child_dtypes != {target_dtype}:
                self._wrap_valid_boundary_or_children(child)

    def _wrap_valid_boundary_or_children(self, module: nn.Module) -> bool:
        """Wrap a valid FSDP boundary, or keep looking below a container."""
        if is_container_module(module):
            return self._wrap_child_boundaries(module)
        return self._wrap_candidate(module)

    def _wrap_child_boundaries(self, module: nn.Module) -> bool:
        """Try to wrap valid FSDP boundaries under this module's direct children."""
        wrapped_any = False
        for child in module.children():
            wrapped_any = self._wrap_valid_boundary_or_children(child) or wrapped_any
        return wrapped_any

    def _safe_fully_shard(
        self,
        module_or_modules: nn.Module | list[nn.Module],
        mp_policy: MixedPrecisionPolicy,
    ) -> bool:
        """Shard one module or one module list as a single FSDP group."""
        modules = self._as_module_group(module_or_modules)
        if not modules or any(id(module) in self.wrapped_module_ids for module in modules):
            return False

        params_before = self._module_group_params(modules)
        dtypes = {param.dtype for param in params_before}
        if len(dtypes) > 1:
            raise ValueError(
                f"FSDP cannot wrap mixed original dtypes {dtypes} in "
                f"{self._module_group_label(modules)}."
            )

        # ``ignored_params`` is what makes nested wrapping safe here. All params
        # already owned by inner groups are excluded from this new group's flat
        # parameter, avoiding duplicate sharding/all-reduce ownership.
        fully_shard_kwargs = dict(self.fsdp_kwargs)
        reshard_after_forward = self._reshard_after_forward_for_group(modules)
        if reshard_after_forward is not None:
            fully_shard_kwargs["reshard_after_forward"] = reshard_after_forward

        fully_shard(
            module_or_modules,
            mp_policy=mp_policy,
            ignored_params=self.wrapped_params,
            **fully_shard_kwargs,
        )
        self._mark_wrapped(modules, params_before)
        return True

    def _as_module_group(self, module_or_modules: nn.Module | list[nn.Module]) -> list[nn.Module]:
        return [module_or_modules] if isinstance(module_or_modules, nn.Module) else list(module_or_modules)

    def _module_group_params(self, modules: list[nn.Module]) -> list[nn.Parameter]:
        params = []
        seen = set()
        for module in modules:
            for param in module_params(module, excluded_param_ids=self.wrapped_param_ids):
                param_id = id(param)
                if param_id in seen:
                    continue
                params.append(param)
                seen.add(param_id)
        return params

    def _module_group_label(self, modules: list[nn.Module]) -> str:
        if len(modules) == 1:
            return modules[0].__class__.__name__
        class_names = {module.__class__.__name__ for module in modules}
        class_label = next(iter(class_names)) if len(class_names) == 1 else "mixed classes"
        return f"{len(modules)} modules ({class_label})"

    def _reshard_after_forward_for_group(self, modules: list[nn.Module]) -> bool | int | None:
        values = [self._reshard_after_forward_for(module) for module in modules]
        first_value = values[0]
        if any(value != first_value for value in values):
            raise ValueError(
                "Cannot create one FSDP group with different "
                f"reshard_after_forward settings: {values}."
            )
        return first_value

    def _mark_wrapped(self, modules: list[nn.Module], params_before: list[nn.Parameter]) -> None:
        """Record module and parameter ownership after a successful fully_shard."""
        self.wrapped_group_count += 1
        for module in modules:
            self.wrapped_module_ids.add(id(module))
        for param in params_before:
            self.wrapped_param_ids.add(id(param))
            self.wrapped_params.add(param)

        # FSDP may replace/register parameter objects during wrapping. Record
        # the module's current recursive parameters as well so later parent
        # groups ignore both original and current parameter objects.
        for module in modules:
            for param in module.parameters(recurse=True):
                self.wrapped_param_ids.add(id(param))
                self.wrapped_params.add(param)
