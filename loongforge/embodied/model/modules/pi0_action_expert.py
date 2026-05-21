"""
Pi0 Action Expert - Pi0/Pi0.5 Independent Action Expert Model

An independent Transformer based on the Gemma architecture, serving as the denoising
network for flow matching. Supports two operating modes:

  Mode 1: Joint Attention (Pi0 original, PaliGemma)
    VLM and Action Expert share attention computation at each layer:
      Layer i:
        Q_vlm, K_vlm, V_vlm = VLM.layer_i.attn(prefix)
        Q_exp, K_exp, V_exp = Expert.layer_i.attn(suffix)
        Q_joint = [Q_vlm; Q_exp], K_joint = [...], V_joint = [...]
        attn_out = Attention(Q_joint, K_joint, V_joint, mask)
        prefix_out, suffix_out = split(attn_out)
        prefix = VLM.layer_i.ffn(prefix_out)   <- separate FFNs
        suffix = Expert.layer_i.ffn(suffix_out) <- separate FFNs

    Pros: Bidirectional information flow (VLM perceives action tokens, Expert perceives image/text tokens)
    Constraints: VLM and Expert must have the same number of layers, and head_dim must be aligned

  Mode 2: Prefix Cache (generic VLM - Qwen/Llama)
    VLM forward once -> project -> feed as prefix to Action Expert self-attention:
      prefix_embs = proj(VLM(images, text).last_hidden_state)
      [prefix; suffix] -> Expert layers -> extract suffix

    Pros: VLM architecture agnostic
    Cons: Unidirectional information flow (VLM -> Expert)

Key components:
  - Pi0ActionExpert: Independent Transformer (18 layers, 300M parameters)
  - _shared_forward(): Layer-wise joint attention (Joint Attention core)
  - Sinusoidal timestep embedding
  - action_in_proj / action_out_proj: action_dim <-> expert_width
  - adaRMS timestep conditioning (Pi0.5 mode)

"""

import logging
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════


def create_sinusoidal_pos_embedding(
    timestep: Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Create sinusoidal positional encoding as timestep embedding.

    Args:
        timestep: (B,) timestep values [0, 1]
        dim: output dimension
        min_period, max_period: sinusoidal period range

    Returns:
        (B, dim) timestep embedding
    """
    if device is None:
        device = timestep.device
    timestep = timestep.to(device)
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)

    half_dim = dim // 2
    freq = torch.exp(
        torch.linspace(
            math.log(min_period), math.log(max_period), half_dim, device=device
        )
    )
    args = timestep[:, None] * (2.0 * math.pi / freq[None, :])
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


def sample_beta(alpha: float, beta: float, batch_size: int, device: torch.device) -> Tensor:
    """Sample from Beta(alpha, beta) distribution."""
    dist = torch.distributions.Beta(alpha, beta)
    return dist.sample((batch_size,)).to(device)


def make_att_2d_masks(
    pad_masks: Tensor,
    att_masks: Tensor,
) -> Tensor:
    """
    Construct 2D attention mask.

    Rules: positions with att_masks==0 (prefix) can be attended to by everyone,
           positions with att_masks==1 (suffix) can only be attended to by prefix + suffix itself.

    Args:
        pad_masks: (B, L) bool
        att_masks: (B, L) -- 0=prefix(bidirectional), 1=suffix

    Returns:
        (B, L, L) bool -- True indicates attention is allowed
    """
    B, L = pad_masks.shape
    base = pad_masks[:, None, :].expand(B, L, L) & pad_masks[:, :, None].expand(B, L, L)
    prefix_mask = (att_masks == 0)
    can_attend_prefix = prefix_mask[:, None, :].expand(B, L, L)
    same_group = (att_masks[:, :, None] == att_masks[:, None, :])
    att_2d = base & (can_attend_prefix | same_group)
    return att_2d


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE (Rotary Position Embedding) to Q and K."""
    # cos, sin: (B, L, head_dim) → expand to (B, 1, L, head_dim) for broadcasting with heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x):
    """Rotary embedding helper: rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ═══════════════════════════════════════════════════════════════
# Expert Transformer Layer (self-contained, no HuggingFace Gemma dependency)
# ═══════════════════════════════════════════════════════════════


class ExpertAttention(nn.Module):
    """Multi-head attention with optional RoPE."""

    def __init__(self, width: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return Q, K, V projections (not attention output)."""
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v


class ExpertMLP(nn.Module):
    """Gated MLP (SwiGLU style)."""

    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """forward"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertTransformerLayer(nn.Module):
    """Single Expert Transformer layer (Pre-Norm + Gated MLP)."""

    def __init__(self, width: int, num_heads: int, head_dim: int, mlp_dim: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(width)
        self.self_attn = ExpertAttention(width, num_heads, head_dim)
        self.post_attention_layernorm = nn.RMSNorm(width)
        self.mlp = ExpertMLP(width, mlp_dim)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Standard pre-norm transformer layer (used in prefix-cache mode)."""
        residual = x
        x_norm = self.input_layernorm(x)
        q, k, v = self.self_attn(x_norm)
        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=self.self_attn.scaling,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.shape)
        x = residual + self.self_attn.o_proj(attn_out)

        residual = x
        x_norm = self.post_attention_layernorm(x)
        x = residual + self.mlp(x_norm)
        return x


# ═══════════════════════════════════════════════════════════════
# Pi0 Action Expert Module
# ═══════════════════════════════════════════════════════════════


class Pi0ActionExpert(nn.Module):
    """
    Pi0/Pi0.5 Action Expert module.

    Contains:
      1. Independent Expert Transformer (usable for joint attention or prefix-cache)
      2. Suffix embedding (action projection + timestep encoding)
      3. Flow matching forward logic
      4. _shared_forward(): Layer-wise joint attention (Joint Attention mode core)
    """

    def __init__(
        self,
        action_dim: int = 7,
        state_dim: int = 7,
        action_horizon: int = 10,
        expert_width: int = 1024,
        expert_depth: int = 18,
        expert_mlp_dim: int = 4096,
        expert_num_heads: int = 8,
        expert_head_dim: int = 128,
        pi05: bool = True,
        num_inference_steps: int = 10,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.expert_width = expert_width
        self.expert_depth = expert_depth
        self.expert_num_heads = expert_num_heads
        self.expert_head_dim = expert_head_dim
        self.pi05 = pi05
        self.num_inference_steps = num_inference_steps
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta

        # ── Projection layers ──
        self.action_in_proj = nn.Linear(action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, action_dim)

        if pi05:
            self.time_mlp_in = nn.Linear(expert_width, expert_width)
            self.time_mlp_out = nn.Linear(expert_width, expert_width)
        else:
            self.state_proj = nn.Linear(state_dim, expert_width)
            self.action_time_mlp_in = nn.Linear(2 * expert_width, expert_width)
            self.action_time_mlp_out = nn.Linear(expert_width, expert_width)

        # ── Expert Transformer Layers (individually accessible, supports joint attention) ──
        self.layers = nn.ModuleList([
            ExpertTransformerLayer(expert_width, expert_num_heads, expert_head_dim, expert_mlp_dim)
            for _ in range(expert_depth)
        ])
        self.norm = nn.RMSNorm(expert_width)

        # ── RoPE (Rotary Position Embedding) ──
        # Precompute cos/sin cache (max sequence length 1024)
        self._init_rope(max_seq_len=1024)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        logger.info(
            f"Pi0ActionExpert: pi05={pi05}, width={expert_width}, depth={expert_depth}, "
            f"heads={expert_num_heads}, head_dim={expert_head_dim}, "
            f"action_dim={action_dim}, horizon={action_horizon}"
        )

    def _init_rope(self, max_seq_len: int = 1024):
        """Initialize Rotary Position Embedding."""
        dim = self.expert_head_dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        cos_cache = freqs.cos()
        sin_cache = freqs.sin()
        # Duplicate for full head_dim: [cos(0), cos(1), ..., cos(0), cos(1), ...]
        self.register_buffer("cos_cache", torch.cat([cos_cache, cos_cache], dim=-1), persistent=False)
        self.register_buffer("sin_cache", torch.cat([sin_cache, sin_cache], dim=-1), persistent=False)

    def _get_rope(self, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get RoPE cos/sin for specified positions.

        Args:
            position_ids: (B, L)

        Returns:
            cos: (B, L, head_dim)
            sin: (B, L, head_dim)
        """
        cos = self.cos_cache[position_ids]  # (B, L, head_dim)
        sin = self.sin_cache[position_ids]
        return cos, sin

    # ═══════════════════════════════════════════════════════════════
    # Suffix Embedding
    # ═══════════════════════════════════════════════════════════════

    def sample_noise(self, shape: Tuple, device: torch.device) -> Tensor:
        """sample noise"""
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """sample time"""
        t_beta = sample_beta(self.noise_beta_alpha, self.noise_beta_beta, batch_size, device)
        t = t_beta * 0.999 + 0.001
        return t.float()

    def embed_suffix(
        self,
        state: Optional[Tensor],
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Construct suffix embedding = [state_emb (optional)] + action_time_emb.

        Returns:
            suffix_embs: (B, suffix_len, expert_width)
            suffix_pad_masks: (B, suffix_len)
            suffix_att_masks: (B, suffix_len) -- 1 indicates suffix group
            adarms_cond: (B, expert_width) or None
        """
        embs = []
        pad_masks = []
        att_mask_vals = []
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device

        if not self.pi05:
            if state is not None:
                if state.dim() == 3:
                    state = state[:, -1, :]
                state_emb = self.state_proj(state.float())
                embs.append(state_emb[:, None, :])
                pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
                att_mask_vals.append(1)

        # Timestep encoding
        time_emb = create_sinusoidal_pos_embedding(timestep, self.expert_width, device=device)
        time_emb = time_emb.to(dtype=noisy_actions.dtype)

        # Action embedding
        action_emb = self.action_in_proj(noisy_actions.float())

        if not self.pi05:
            time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)
            action_time = torch.cat([action_emb, time_emb_expanded], dim=-1)
            x = self.action_time_mlp_in(action_time)
            x = F.silu(x)
            action_time_emb = self.action_time_mlp_out(x)
            adarms_cond = None
        else:
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            adarms_cond = F.silu(x)
            action_time_emb = action_emb

        embs.append(action_time_emb)
        T = action_time_emb.shape[1]
        pad_masks.append(torch.ones(bsize, T, dtype=torch.bool, device=device))
        att_mask_vals.extend([1] * T)

        suffix_embs = torch.cat(embs, dim=1)
        suffix_pad_masks = torch.cat(pad_masks, dim=1)
        suffix_att_masks = torch.tensor(
            att_mask_vals, dtype=suffix_embs.dtype, device=device
        )[None, :].expand(bsize, -1)

        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond

    # ═══════════════════════════════════════════════════════════════
    # Mode 1: Joint Attention (_shared_forward)
    #
    # This is the core innovation of Pi0: VLM and Expert share attention layer by layer.
    # Both models compute their own Q/K/V, concatenate for joint attention,
    # then split back and go through their respective FFNs.
    # ═══════════════════════════════════════════════════════════════

    def _shared_forward(
        self,
        vlm_language_model: nn.Module,
        prefix_embs: Tensor,
        suffix_embs: Tensor,
        attention_mask_4d: Tensor,
        position_ids: Tensor,
        adarms_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Layer-wise joint attention forward (Joint Attention).

        Core algorithm:
          for each layer i:
            1. VLM.layer_i: prefix -> LayerNorm -> Q_vlm, K_vlm, V_vlm
            2. Expert.layer_i: suffix -> LayerNorm -> Q_exp, K_exp, V_exp
            3. Joint: Q = [Q_vlm; Q_exp], K = [K_vlm; K_exp], V = [V_vlm; V_exp]
            4. Apply RoPE to Q, K
            5. attn_out = scaled_dot_product_attention(Q, K, V, mask)
            6. Split: attn_vlm, attn_exp = split(attn_out)
            7. prefix = prefix + VLM.layer_i.o_proj(attn_vlm)
            8. suffix = suffix + Expert.layer_i.o_proj(attn_exp)
            9. prefix = prefix + VLM.layer_i.mlp(VLM.layer_i.post_norm(prefix))
            10. suffix = suffix + Expert.layer_i.mlp(Expert.layer_i.post_norm(suffix))

        Args:
            vlm_language_model: VLM's language model (must have .layers[i] attribute)
                Must satisfy the interface:
                  vlm_language_model.layers[i].input_layernorm(x) -> normed
                  vlm_language_model.layers[i].self_attn.{q,k,v,o}_proj
                  vlm_language_model.layers[i].post_attention_layernorm(x) -> normed
                  vlm_language_model.layers[i].mlp(x)
            prefix_embs: (B, L_prefix, W) VLM prefix embeddings
            suffix_embs: (B, L_suffix, W) action suffix embeddings
            attention_mask_4d: (B, 1, L_total, L_total) float mask (-inf for blocked)
            position_ids: (B, L_total) position indices
            adarms_cond: (B, W) optional timestep conditioning for suffix layers

        Returns:
            suffix_out: (B, L_suffix, W) -- only returns suffix output
        """
        prefix_len = prefix_embs.shape[1]
        suffix_len = suffix_embs.shape[1]
        num_layers = min(len(self.layers), self._count_vlm_layers(vlm_language_model))

        # Get RoPE
        cos, sin = self._get_rope(position_ids)

        for layer_idx in range(num_layers):
            vlm_layer = self._get_vlm_layer(vlm_language_model, layer_idx)
            expert_layer = self.layers[layer_idx]

            def _compute_joint_layer(
                layer_idx, prefix_embs, suffix_embs, attention_mask_4d, cos, sin, adarms_cond
            ):
                vlm_l = self._get_vlm_layer(vlm_language_model, layer_idx)
                exp_l = self.layers[layer_idx]

                # ── Step 1-2: Respective LayerNorm + QKV ──
                prefix_normed = vlm_l.input_layernorm(prefix_embs)
                suffix_normed = exp_l.input_layernorm(suffix_embs)

                # VLM Q/K/V
                B = prefix_embs.shape[0]
                q_vlm, k_vlm, v_vlm = vlm_l.self_attn(prefix_normed)
                # Expert Q/K/V
                q_exp, k_exp, v_exp = exp_l.self_attn(suffix_normed)

                # ── Step 3: Concat for joint attention ──
                q = torch.cat([q_vlm, q_exp], dim=2)  # (B, heads, L_total, head_dim)
                k = torch.cat([k_vlm, k_exp], dim=2)
                v = torch.cat([v_vlm, v_exp], dim=2)

                # ── Step 4: Apply RoPE ──
                # cos/sin: (B, L_total, head_dim) → need (B, 1, L_total, head_dim)
                cos_expanded = cos.unsqueeze(1)
                sin_expanded = sin.unsqueeze(1)
                q = (q * cos_expanded) + (_rotate_half(q) * sin_expanded)
                k = (k * cos_expanded) + (_rotate_half(k) * sin_expanded)

                # ── Step 5: Scaled dot-product attention ──
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask_4d,
                    scale=exp_l.self_attn.scaling,
                )
                # attn_out: (B, heads, L_total, head_dim)

                # ── Step 6: Split ──
                attn_out = attn_out.transpose(1, 2).contiguous()
                # (B, L_total, heads*head_dim)
                total_head_dim = attn_out.shape[-1]
                attn_out = attn_out.view(B, prefix_len + suffix_len, total_head_dim)
                attn_prefix = attn_out[:, :prefix_len, :]
                attn_suffix = attn_out[:, prefix_len:, :]

                # ── Step 7-8: o_proj + residual ──
                prefix_embs = prefix_embs + vlm_l.self_attn.o_proj(attn_prefix)
                suffix_embs = suffix_embs + exp_l.self_attn.o_proj(attn_suffix)

                # ── Step 9-10: Post-norm + MLP + residual ──
                prefix_normed2 = vlm_l.post_attention_layernorm(prefix_embs)
                prefix_embs = prefix_embs + vlm_l.mlp(prefix_normed2)

                suffix_normed2 = exp_l.post_attention_layernorm(suffix_embs)
                suffix_embs = suffix_embs + exp_l.mlp(suffix_normed2)

                return prefix_embs, suffix_embs

            # Optional gradient checkpointing
            if self.gradient_checkpointing and self.training:
                prefix_embs, suffix_embs = grad_checkpoint(
                    _compute_joint_layer,
                    layer_idx, prefix_embs, suffix_embs,
                    attention_mask_4d, cos, sin, adarms_cond,
                    use_reentrant=False,
                )
            else:
                prefix_embs, suffix_embs = _compute_joint_layer(
                    layer_idx, prefix_embs, suffix_embs,
                    attention_mask_4d, cos, sin, adarms_cond,
                )

        # Final norm on suffix only
        suffix_out = self.norm(suffix_embs)
        return suffix_out

    def _count_vlm_layers(self, vlm_language_model: nn.Module) -> int:
        """Get the number of layers in the VLM language model."""
        if hasattr(vlm_language_model, 'layers'):
            return len(vlm_language_model.layers)
        if hasattr(vlm_language_model, 'config') and hasattr(vlm_language_model.config, 'num_hidden_layers'):
            return vlm_language_model.config.num_hidden_layers
        return self.expert_depth

    def _get_vlm_layer(self, vlm_language_model: nn.Module, idx: int):
        """Get the i-th layer of the VLM."""
        if hasattr(vlm_language_model, 'layers'):
            return vlm_language_model.layers[idx]
        raise AttributeError(
            "vlm_language_model must have .layers attribute for joint attention mode"
        )

    # ═══════════════════════════════════════════════════════════════
    # Mode 2: Prefix Cache (independent Expert forward)
    # ═══════════════════════════════════════════════════════════════

    def expert_forward(
        self,
        all_embs: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Action Expert independent forward (prefix-cache mode).

        Concatenates [prefix; suffix] and passes through all Expert layers.
        VLM does not participate in computation (its output has been projected as prefix embeddings).

        Args:
            all_embs: (B, L, W)
            attention_mask: (B, 1, L, L) float mask or None

        Returns:
            (B, L, W)
        """
        x = all_embs
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        return x

    # ═══════════════════════════════════════════════════════════════
    # Training: compute_loss (supports both modes)
    # ═══════════════════════════════════════════════════════════════

    def compute_loss(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        target_actions: Tensor,
        state: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        vlm_language_model: Optional[nn.Module] = None,
    ) -> Tensor:
        """
        Compute flow matching training loss.

        Automatic mode selection:
          - If vlm_language_model is provided -> Joint Attention (_shared_forward)
          - Otherwise -> Prefix Cache (expert_forward)

        Args:
            prefix_embs: (B, L_prefix, W)
            prefix_pad_masks: (B, L_prefix)
            target_actions: (B, action_horizon, action_dim)
            state: (B, state_dim) optional
            vlm_language_model: VLM LM (with .layers attribute) for joint attention
        """
        B = target_actions.shape[0]
        device = target_actions.device

        if noise is None:
            noise = self.sample_noise(target_actions.shape, device)
        if time is None:
            time = self.sample_time(B, device)

        # Flow matching interpolation
        t_expand = time[:, None, None]
        x_t = (1.0 - t_expand) * target_actions + t_expand * noise
        u_t = noise - target_actions

        # Build suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )

        # Dtype alignment
        if prefix_embs.dtype != suffix_embs.dtype:
            suffix_embs = suffix_embs.to(dtype=prefix_embs.dtype)

        if vlm_language_model is not None:
            # ═══ Mode 1: Joint Attention ═══
            # Build combined masks
            prefix_att_masks = torch.zeros(
                B, prefix_embs.shape[1], dtype=suffix_embs.dtype, device=device
            )
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1

            # Convert to 4D float mask: True → 0, False → -inf
            att_4d = att_2d[:, None, :, :]
            att_4d = torch.where(att_4d, 0.0, -2.3819763e38).to(dtype=prefix_embs.dtype)

            # _shared_forward: layer-wise joint attention
            suffix_out = self._shared_forward(
                vlm_language_model,
                prefix_embs, suffix_embs,
                att_4d, position_ids,
                adarms_cond,
            )
        else:
            # ═══ Mode 2: Prefix Cache ═══
            all_embs = torch.cat([prefix_embs, suffix_embs], dim=1)

            # Build attention mask
            prefix_att_masks = torch.zeros(
                B, prefix_embs.shape[1], dtype=suffix_embs.dtype, device=device
            )
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d = make_att_2d_masks(pad_masks, att_masks)
            att_4d = att_2d[:, None, :, :]
            att_4d = torch.where(att_4d, 0.0, -2.3819763e38).to(dtype=prefix_embs.dtype)

            expert_out = self.expert_forward(all_embs, attention_mask=att_4d)
            suffix_len = suffix_pad_masks.shape[1]
            suffix_out = expert_out[:, -suffix_len:]

        # Extract action tokens and project
        suffix_out = suffix_out[:, -self.action_horizon:]
        suffix_out = suffix_out.float()
        v_t = self.action_out_proj(suffix_out)

        # MSE loss
        loss = F.mse_loss(v_t, u_t, reduction="none")
        return loss

    # ═══════════════════════════════════════════════════════════════
    # Inference: sample_actions (supports both modes)
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def sample_actions(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        state: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        vlm_language_model: Optional[nn.Module] = None,
    ) -> Tensor:
        """
        Euler ODE inference: iterative denoising from pure noise.

        Automatic mode selection:
          - If vlm_language_model is provided -> Joint Attention (recompute each step)
          - Otherwise -> Prefix Cache (prefix passes through expert once, subsequent KV cache reuse)

        Returns:
            (B, action_horizon, action_dim) predicted actions
        """
        num_steps = num_steps or self.num_inference_steps
        B = prefix_embs.shape[0]
        device = prefix_embs.device

        x_t = self.sample_noise((B, self.action_horizon, self.action_dim), device)
        dt = -1.0 / num_steps
        time_val = 1.0

        for _ in range(num_steps):
            t = torch.full((B,), time_val, dtype=torch.float32, device=device)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, t
            )

            if prefix_embs.dtype != suffix_embs.dtype:
                suffix_embs = suffix_embs.to(dtype=prefix_embs.dtype)

            if vlm_language_model is not None:
                # Joint Attention mode (requires full recomputation each step)
                prefix_att_masks = torch.zeros(
                    B, prefix_embs.shape[1], dtype=suffix_embs.dtype, device=device
                )
                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
                att_2d = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1
                att_4d = att_2d[:, None, :, :]
                att_4d = torch.where(att_4d, 0.0, -2.3819763e38).to(dtype=prefix_embs.dtype)

                suffix_out = self._shared_forward(
                    vlm_language_model,
                    prefix_embs, suffix_embs,
                    att_4d, position_ids,
                    adarms_cond,
                )
            else:
                # Prefix-cache mode
                all_embs = torch.cat([prefix_embs, suffix_embs], dim=1)
                prefix_att_masks = torch.zeros(
                    B, prefix_embs.shape[1], dtype=suffix_embs.dtype, device=device
                )
                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
                att_2d = make_att_2d_masks(pad_masks, att_masks)
                att_4d = att_2d[:, None, :, :]
                att_4d = torch.where(att_4d, 0.0, -2.3819763e38).to(dtype=prefix_embs.dtype)

                expert_out = self.expert_forward(all_embs, attention_mask=att_4d)
                suffix_len = suffix_pad_masks.shape[1]
                suffix_out = expert_out[:, -suffix_len:]

            # Extract and project
            suffix_out = suffix_out[:, -self.action_horizon:]
            suffix_out = suffix_out.float()
            v_t = self.action_out_proj(suffix_out)

            # Euler step
            x_t = x_t + v_t * dt
            time_val += dt

        return x_t
