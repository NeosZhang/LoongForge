# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""warmup tools for dp balance"""

import torch
import torch.distributed as dist

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Callable

from megatron.training import get_args
from megatron.core import mpu
from loongforge.utils import constants


# ============================================================================
# Shared utilities
# ============================================================================

def _softmax_max(vals, alpha=20):
    """
    Smooth approximation of max(), used to model the DP synchronization cost
    dominated by the slowest rank.
    """
    vals = np.array(vals, dtype=float)
    m = vals.max()
    return m + (1 / alpha) * np.log(np.sum(np.exp(alpha * (vals - m))))


@dataclass
class _WarmupConfig:
    """Configuration for warmup coefficient fitting.

    This class encapsulates the differences between VLM and ViT warmup
    while providing a common interface for coefficient fitting.
    """
    # Name for logging
    name: str

    # Enable attribute name (e.g., 'use_vlm_dp_balance', 'use_vit_dp_balance')
    enable_attr: str

    # Warmup iteration attribute name
    iters_attr: str

    # Warmup state object (holds profiling data and coefficients)
    state: '_WarmupState'

    # Scaling factors for optimization stability (list of scales for each coefficient term)
    scales: List[float] = field(default_factory=list)

    # Initial guess for optimization
    init_vals: Tuple = field(default_factory=tuple)

    # Loss function that computes predicted load from coefficients and vars
    loss_fn: Callable = field(default_factory=lambda: None)


@dataclass
class _WarmupState:
    """Base class for warmup state management.

    Stores profiling data and computation coefficients for either VLM or ViT.
    """
    # Profiling buffers
    var_groups: List = field(default_factory=list)
    forward_time: List = field(default_factory=list)


@dataclass
class _VLMWarmupState(_WarmupState):
    """Warmup state for VLM (Vision-Language Model).

    VLM cost model: load ≈ a * seq_len² + b * seq_len + c * seq_num
    """
    # Coefficients for the cost model
    seq2_coef: float = 0.0  # quadratic term
    seq_coef: float = 1.0   # linear term
    seq_num_coef: float = 0.0  # per-sample overhead

    @property
    def coef_tuple(self) -> Tuple[float, float, float]:
        """Return coefficients as a tuple."""
        return self.seq2_coef, self.seq_coef, self.seq_num_coef

    def set_coefs(self, seq2_coef: float, seq_coef: float, seq_num_coef: float):
        """Set all coefficients."""
        self.seq2_coef = seq2_coef
        self.seq_coef = seq_coef
        self.seq_num_coef = seq_num_coef


@dataclass
class _ViTWarmupState(_WarmupState):
    """Warmup state for ViT (Vision Transformer).

    ViT cost model: load ≈ a * num_patches² + b * num_patches + c * num_images
    """
    # Coefficients for the cost model
    num_patches_sq_coef: float = 0.0  # quadratic term (attention)
    num_patches_coef: float = 1.0   # linear term (MLP, LayerNorm, etc.)
    num_images_coef: float = 0.0  # per-image overhead (CLS, pos embed, etc.)

    @property
    def coef_tuple(self) -> Tuple[float, float, float]:
        """Return coefficients as a tuple."""
        return self.num_patches_sq_coef, self.num_patches_coef, self.num_images_coef

    def set_coefs(self, num_patches_sq_coef: float, num_patches_coef: float, num_images_coef: float):
        """Set all coefficients."""
        self.num_patches_sq_coef = num_patches_sq_coef
        self.num_patches_coef = num_patches_coef
        self.num_images_coef = num_images_coef


# Global warmup states
_VLM_STATE = _VLMWarmupState()
_VIT_STATE = _ViTWarmupState()


def _set_warmup_c1_generic(
    state: _WarmupState,
    config: _WarmupConfig,
    c1: float,
):
    """
    Generic function to record forward latency during warm-up.

    Args:
        state: WarmupState object to append time to.
        config: WarmupConfig containing enable_attr and iters_attr.
        c1: Measured forward computation time.
    """
    args_train = get_args()
    iteration = args_train.curr_iteration

    # Check if we should record (use hasattr for ViT which may not have attrs in all contexts)
    has_enable = hasattr(args_train, config.enable_attr)
    if not has_enable or not getattr(args_train, config.enable_attr, False):
        return

    has_iters = hasattr(args_train, config.iters_attr)
    if not has_iters:
        return

    iters = getattr(args_train, config.iters_attr, [])
    if iteration not in iters or iteration == iters[0]:
        return

    state.forward_time.append(c1)


def _set_warmup_groups_generic_vit(
    state: _WarmupState,
    config: _WarmupConfig,
    input_lengths: torch.Tensor,
):
    """
    Generic function to collect per-DP input statistics during warm-up (ViT mode).

    For the current iteration, computes:
        - sum(input_length²)
        - sum(input_length)
        - number of samples (images)

    across all samples on each DP rank, gathers them across all DP ranks,
    and stores the result.

    Args:
        state: WarmupState object to append data to.
        config: WarmupConfig containing enable_attr and iters_attr.
        input_lengths: Tensor of shape [num_samples] containing the input length
                     for each sample (e.g., num_patches for ViT).
    """
    args_train = get_args()
    iteration = args_train.curr_iteration

    # Check if we should collect
    has_enable = hasattr(args_train, config.enable_attr)
    if not has_enable or not getattr(args_train, config.enable_attr, False):
        return

    has_iters = hasattr(args_train, config.iters_attr)
    if not has_iters:
        return

    iters = getattr(args_train, config.iters_attr, [])
    if iteration not in iters or iteration == iters[0]:
        return

    dp_group = mpu.get_data_parallel_group_gloo(
        with_context_parallel=False,
        partial_data_parallel=False,
    )
    dp_size = dp_group.size()

    # Compute statistics for this DP rank
    num_samples = input_lengths.numel()
    input_length_sum = input_lengths.sum().item()
    input_length_sq_sum = (input_lengths ** 2).sum().item()

    # Prepare all-gather buffers
    num_samples_tensor = torch.tensor(
        [num_samples],
        device=input_lengths.device,
        dtype=torch.long,
    )
    length_sum_tensor = torch.tensor(
        [input_length_sum],
        device=input_lengths.device,
        dtype=torch.float,
    )
    length_sq_tensor = torch.tensor(
        [input_length_sq_sum],
        device=input_lengths.device,
        dtype=torch.float,
    )

    num_samples_list = [torch.zeros_like(num_samples_tensor) for _ in range(dp_size)]
    length_sum_list = [torch.zeros_like(length_sum_tensor) for _ in range(dp_size)]
    length_sq_list = [torch.zeros_like(length_sq_tensor) for _ in range(dp_size)]

    # Gather statistics across DP ranks
    dist.all_gather(num_samples_list, num_samples_tensor, group=dp_group)
    dist.all_gather(length_sum_list, length_sum_tensor, group=dp_group)
    dist.all_gather(length_sq_list, length_sq_tensor, group=dp_group)

    dp_rank = mpu.get_data_parallel_rank()
    is_dp_root = dp_rank == 0

    # Convert tensors to Python scalars
    num_samples_list = [t.item() for t in num_samples_list]
    length_sum_list = [t.item() for t in length_sum_list]
    length_sq_list = [t.item() for t in length_sq_list]

    # Store per-DP variable group: (sum(length²), sum(length), num_samples)
    var_group = [
        (a, b, c)
        for (a, b, c) in zip(length_sq_list, length_sum_list, num_samples_list)
    ]
    if is_dp_root:
        state.var_groups.append(var_group)


def _solve_computation_coef_generic(config: _WarmupConfig) -> bool:
    """
    Generic coefficient fitting function that works for both VLM and ViT.

    Args:
        config: WarmupConfig containing all parameters and callbacks.

    Returns:
        bool: True if fitting was performed, False otherwise.
    """
    state = config.state

    args_train = get_args()
    iteration = args_train.curr_iteration
    dp_rank = mpu.get_data_parallel_rank()
    is_dp_root = dp_rank == 0

    # Only run during DP-balance warm-up phase
    has_enable = hasattr(args_train, config.enable_attr)
    if (
        not is_dp_root
        or not has_enable
        or not getattr(args_train, config.enable_attr, False)
        or not hasattr(args_train, config.iters_attr)
        or iteration != getattr(args_train, config.iters_attr)[-1] + 1
    ):
        return False

    # Warm-up profiling data
    var_groups_data, C = state.var_groups, state.forward_time

    def loss(vars, var_groups_data, C):
        """
        Objective function:
        Minimize the squared error between predicted max DP load and
        observed forward latency.
        """
        err = 0.0

        for terms, Ci in zip(var_groups_data, C):
            # Predicted per-DP loads for this iteration
            vals = config.loss_fn(vars, terms)
            err += (_softmax_max(vals) - Ci) ** 2

        return err

    # Coefficients are constrained to be non-negative
    bounds = [(0, None)] * len(config.init_vals)

    # Solve the nonlinear least-squares problem
    res = minimize(loss, config.init_vals, args=(var_groups_data, C), bounds=bounds)

    # Update coefficient in state using model-specific setter
    if len(config.scales) > 1:
        # Multi-coefficient case (VLM): scale back to original
        scaled_coefs = [res.x[i] / config.scales[i] for i in range(len(res.x))]
        state.set_coefs(*scaled_coefs)
    else:
        # Single-coefficient case (ViT): use directly
        state.set_coef(res.x[0])

    return True


# ============================================================================
# VLM (text+vision) warmup
# ============================================================================

def _vlm_loss_fn(vars, terms):
    """Compute predicted load for VLM model."""
    x_t, y_t, z_t = vars
    S_a, S_b, S_c = 1e8, 1e4, 1e1
    return [
        (a / S_a) * x_t + (b / S_b) * y_t + (c / S_c) * z_t
        for (a, b, c) in terms
    ]


# VLM warmup configuration
_VLM_WARMUP_CONFIG = _WarmupConfig(
    name="VLM",
    enable_attr="use_vlm_dp_balance",
    iters_attr="vlm_dp_balance_warmup_iters",
    state=_VLM_STATE,
    scales=[1e8, 1e4, 1e1],
    init_vals=(0, 0, 0),
    loss_fn=_vlm_loss_fn,
)


def get_vlm_seq_coefs():
    """Get the coefficients of the VLM attention computation cost model.

    Returns:
        Tuple[float, float, float]:
            - Quadratic coefficient (VLM_SEQ2_COEF): coefficient for sequence length squared term
            - Linear coefficient (VLM_SEQ_COEF): coefficient for sequence length linear term
            - Constant coefficient (VLM_SEQ_NUM_COEF): fixed overhead coefficient per sequence
    """
    return _VLM_STATE.coef_tuple


def set_vlm_seq_coefs(seq2_coef: float, seq_coef: float, seq_num_coef: float):
    """Set the coefficients of the VLM attention computation cost model.

    Args:
        seq2_coef: Coefficient for sequence length squared term.
        seq_coef: Coefficient for sequence length linear term.
        seq_num_coef: Fixed overhead coefficient per sequence.
    """
    _VLM_STATE.set_coefs(seq2_coef, seq_coef, seq_num_coef)


def solve_vlm_computation_coef():
    """
    Fit the VLM attention computation cost model from warm-up profiling data.

    The per-pack compute load is modeled as:
        load ≈ a * seq_len² + b * seq_len + c * seq_num

    Coefficients (a, b, c) are estimated by minimizing the squared error
    between predicted and measured forward latency during warm-up.
    """
    return _solve_computation_coef_generic(_VLM_WARMUP_CONFIG)


def set_vlm_warmup_c1(c1: float):
    """
    Record VLM forward latency during the DP-balance warm-up phase.

    This function appends the measured VLM forward computation time `c1`
    for the current iteration, which is later used to fit the VLM
    computation cost model.

    Args:
        c1: Measured forward computation time in milliseconds.
    """
    _set_warmup_c1_generic(_VLM_STATE, _VLM_WARMUP_CONFIG, c1)


def set_vlm_warmup_groups(data: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]):
    """
    Collect per-DP VLM sequence statistics during the warm-up phase.

    For the current iteration, this function computes:
        - sum(seq_len²)
        - sum(seq_len)
        - number of sequences

    across all micro-batches on each DP rank, gathers them across all DP
    ranks, and stores the resulting per-DP variable group. These statistics
    are later used to fit the DP computation cost model.

    Args:
        data: A single micro-batch dict, or a list of micro-batch dicts
              (one per micro-batch in the iteration).
    """
    state = _VLM_STATE
    args_train = get_args()
    iteration = args_train.curr_iteration

    if (
        not args_train.use_vlm_dp_balance
        or iteration not in args_train.vlm_dp_balance_warmup_iters
        or iteration == args_train.vlm_dp_balance_warmup_iters[0]
    ):
        return

    # Normalize to list for uniform handling
    if isinstance(data, dict):
        data_list = [data]
    else:
        data_list = data

    dp_group = mpu.get_data_parallel_group_gloo(
        with_context_parallel=False,
        partial_data_parallel=False,
    )
    dp_size = dp_group.size()

    # Accumulate statistics across all micro-batches
    total_seq_num = 0
    total_seq_lenth_sum = None
    total_seq_lenth_square_sum = None

    for micro_batch in data_list:
        cu_lengths = None
        if args_train.model_family == "intern_vl":
            cu_lengths = micro_batch["attn_mask"]
        elif args_train.model_family in constants.VisionLanguageModelFamilies.names():
            cu_lengths = micro_batch["cu_lengths"]
        cu_lengths = cu_lengths.squeeze(0)

        # Number of sequences in this micro-batch
        seq_num = cu_lengths.numel() - 1
        total_seq_num += seq_num

        # Per-sequence lengths and their squared values
        seq_lenth = cu_lengths[1:] - cu_lengths[:-1]
        seq_lenth_square = seq_lenth**2

        if total_seq_lenth_sum is None:
            total_seq_lenth_sum = seq_lenth.sum()
            total_seq_lenth_square_sum = seq_lenth_square.sum()
        else:
            total_seq_lenth_sum = total_seq_lenth_sum + seq_lenth.sum()
            total_seq_lenth_square_sum = total_seq_lenth_square_sum + seq_lenth_square.sum()

    seq_num_tensor = torch.tensor(
        [total_seq_num],
        device=total_seq_lenth_sum.device,
        dtype=torch.long,
    )

    # Prepare all-gather buffers
    seq_num_list = [torch.zeros_like(seq_num_tensor) for _ in range(dp_size)]
    seq_lenth_sum_list = [torch.zeros_like(total_seq_lenth_sum) for _ in range(dp_size)]
    seq_lenth_square_sum_list = [
        torch.zeros_like(total_seq_lenth_square_sum) for _ in range(dp_size)
    ]

    # Gather statistics across DP ranks
    dist.all_gather(seq_num_list, seq_num_tensor, group=dp_group)
    dist.all_gather(seq_lenth_sum_list, total_seq_lenth_sum, group=dp_group)
    dist.all_gather(seq_lenth_square_sum_list, total_seq_lenth_square_sum, group=dp_group)

    dp_rank = mpu.get_data_parallel_rank()
    is_dp_root = dp_rank == 0

    # Convert tensors to Python scalars
    seq_lenth_square_sum_list = [t.item() for t in seq_lenth_square_sum_list]
    seq_lenth_sum_list = [t.item() for t in seq_lenth_sum_list]
    seq_num_list = [t.item() for t in seq_num_list]

    # Store per-DP variable group: (sum(seq_len²), sum(seq_len), seq_num)
    var_group = [
        (a, b, c)
        for (a, b, c) in zip(
            seq_lenth_square_sum_list,
            seq_lenth_sum_list,
            seq_num_list,
        )
    ]
    if is_dp_root:
        state.var_groups.append(var_group)


def load_estimate_per_vlm_sample(seq_len: int) -> float:
    """
    Estimate the relative computation load of a single VLM sample.

    Cost model:
        load ≈ a * seq_len² + b * seq_len + c

    This estimate is used for VLM sample reordering and DP load balancing.
    Coefficients are calibrated during the VLM warm-up phase.

    Args:
        seq_len: Sequence length of the sample.

    Returns:
        Estimated computation load.
    """
    return _VLM_STATE.seq2_coef * seq_len**2 + _VLM_STATE.seq_coef * seq_len + _VLM_STATE.seq_num_coef


# ============================================================================
# ViT encoder warmup
# ============================================================================

def _vit_loss_fn(vars, terms):
    """Compute predicted load for ViT model.

    Terms format: (sum(num_patches²), sum(num_patches), num_images)
    """
    x_t, y_t, z_t = vars
    S_a, S_b, S_c = 1e8, 1e4, 1e1
    return [
        (a / S_a) * x_t + (b / S_b) * y_t + (c / S_c) * z_t
        for (a, b, c) in terms
    ]


# ViT warmup configuration
_VIT_WARMUP_CONFIG = _WarmupConfig(
    name="ViT",
    enable_attr="use_vit_dp_balance",
    iters_attr="vit_dp_balance_warmup_iters",
    state=_VIT_STATE,
    scales=[1e8, 1e4, 1e1],
    init_vals=(0, 0, 0),
    loss_fn=_vit_loss_fn,
)


def get_vit_computation_coef() -> Tuple[float, float, float]:
    """Get the coefficients of the ViT computation cost model.

    Returns:
        Tuple[float, float, float]:
            - Quadratic coefficient: coefficient for num_patches squared term (attention)
            - Linear coefficient: coefficient for num_patches linear term (MLP, LayerNorm, etc.)
            - Constant coefficient: per-image overhead (CLS, pos embed, etc.)
    """
    return _VIT_STATE.coef_tuple


def set_vit_computation_coef(num_patches_sq_coef: float, num_patches_coef: float, num_images_coef: float):
    """Set the coefficients of the ViT computation cost model.

    Args:
        num_patches_sq_coef: Coefficient for num_patches squared term.
        num_patches_coef: Coefficient for num_patches linear term.
        num_images_coef: Per-image overhead coefficient.
    """
    _VIT_STATE.set_coefs(num_patches_sq_coef, num_patches_coef, num_images_coef)


def solve_vit_computation_coef():
    """
    Fit the ViT computation cost model from warm-up profiling data.

    The per-DP compute load is modeled as:
        load ≈ a * num_patches² + b * num_patches + c * num_images

    Coefficients (a, b, c) are estimated by minimizing the squared error
    between predicted and measured ViT forward latency during warm-up.
    """
    return _solve_computation_coef_generic(_VIT_WARMUP_CONFIG)


def set_vit_warmup_c1(c1: float):
    """
    Record ViT forward latency during the DP-balance warm-up phase.

    This function appends the measured ViT forward computation time `c1`
    for the current iteration, which is later used to fit the ViT
    computation cost model.

    Args:
        c1: Measured forward computation time in milliseconds.
    """
    _set_warmup_c1_generic(_VIT_STATE, _VIT_WARMUP_CONFIG, c1)


def set_vit_warmup_groups(vit_input_lengths: torch.Tensor):
    """
    Collect per-DP ViT input statistics during the warm-up phase.

    For the current iteration, this function computes:
        - sum(num_patches²)
        - sum(num_patches)
        - number of images

    across all images on each DP rank, gathers them across all DP
    ranks, and stores the resulting per-DP variable group. These
    statistics are later used to fit the ViT computation cost model.

    Args:
        vit_input_lengths: Tensor of shape [num_images] containing
            the input length (num_patches) for each image.
    """
    _set_warmup_groups_generic_vit(_VIT_STATE, _VIT_WARMUP_CONFIG, vit_input_lengths)


def load_estimate_per_vit_sample(vit_input_length: int) -> float:
    """
    Estimate the relative computation load of a single image for ViT.

    Cost model:
        load ≈ a * num_patches² + b * num_patches + c

    This estimate is used for ViT DP load balancing.
    The coefficients are calibrated during the warm-up phase.

    Args:
        vit_input_length: Input length (num_patches) of the image.

    Returns:
        Estimated computation load.
    """
    return (
        _VIT_STATE.num_patches_sq_coef * vit_input_length ** 2
        + _VIT_STATE.num_patches_coef * vit_input_length
        + _VIT_STATE.num_images_coef
    )


# ============================================================================
# Backward compatibility: expose globals as before for any external access
# ============================================================================

# VLM globals (for backward compatibility)
VLM_WARMUP_VAR_GROUPS = _VLM_STATE.var_groups
VLM_WARM_FORWARD_TIME = _VLM_STATE.forward_time
VLM_SEQ2_COEF = _VLM_STATE.seq2_coef
VLM_SEQ_COEF = _VLM_STATE.seq_coef
VLM_SEQ_NUM_COEF = _VLM_STATE.seq_num_coef

# ViT globals (for backward compatibility)
VIT_WARMUP_VAR_GROUPS = _VIT_STATE.var_groups
VIT_WARM_FORWARD_TIME = _VIT_STATE.forward_time
VIT_NUM_PATCHES_SQ_COEF = _VIT_STATE.num_patches_sq_coef
VIT_NUM_PATCHES_COEF = _VIT_STATE.num_patches_coef
VIT_NUM_IMAGES_COEF = _VIT_STATE.num_images_coef
