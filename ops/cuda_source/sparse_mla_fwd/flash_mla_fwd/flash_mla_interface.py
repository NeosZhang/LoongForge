"""
flash mla interface
"""
from typing import Optional, Tuple

import torch

import flash_mla_fwd.cuda as flash_mla_cuda


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    q_start_index_s: int = 0,
    write_p_out: bool = False,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    window_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        q_start_index_s: The starting position of the current chunk in the global sequence (used for causal masking)
        write_p_out: bool. Whether to write p_out to global memory.
        topk_length: optional, [s_q], int32. Per-query valid topk length.
        attn_sink: optional, [h_q], float32. Per-head attention sink logit in score space.
            If provided, output is scaled by exp(lse) / (exp(lse) + exp(attn_sink)).
            -inf has no effect; +inf zeros out the corresponding head output.
        window_size: int. Prefix topk length for sliding-window entries. Can be > 0 only when write_p_out=True.

    Returns:
        (output, max_logits, lse, p_out)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
        - p_out: [s_q, h_q, topk - window_size], float32, probability（write_p_out=False 时为None）
    """
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, q_start_index_s, write_p_out, topk_length, attn_sink, window_size
    )
    return results

