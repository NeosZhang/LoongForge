"""
Pi0JointAttention - Pi0/Pi0.5 Joint Attention Alignment Strategy

Core algorithm (two modes):

Mode 1: Joint Attention (PaliGemma native)
  VLM and Action Expert share attention computation at each layer:
    Layer i:
      Q_vlm, K_vlm, V_vlm = VLM.layer_i(prefix_embs)
      Q_exp, K_exp, V_exp = Expert.layer_i(suffix_embs)
      Q_joint = [Q_vlm; Q_exp], K_joint = [K_vlm; K_exp], V_joint = [V_vlm; V_exp]
      attn_out = Attention(Q_joint, K_joint, V_joint)
      vlm_out, exp_out = split(attn_out)
      prefix_embs = VLM.ffn_i(vlm_out)
      suffix_embs = Expert.ffn_i(exp_out)

  Pros: Bidirectional information flow, VLM can perceive action token states
  Cons: Requires VLM and Expert to have the same number of layers, inference requires full recomputation

Mode 2: Prefix Cache (generic VLM - Qwen/Llama)
  VLM forward once -> project -> feed as prefix to Action Expert:
    prefix_embs = project(VLM(images, text).last_hidden_state)
    suffix_out = ActionExpert([prefix_embs; suffix_embs], prefix_kv_cache=None)

  Pros: VLM architecture agnostic (only needs to output hidden states)
  Cons: Unidirectional information flow (VLM -> Expert, no Expert -> VLM feedback)

Corresponding models: Pi0, Pi0.5, LlamaPi0
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from model.compose.registry import CONDITION_REGISTRY
from model.compose.condition.base import BaseCondition


@CONDITION_REGISTRY.register("Pi0JointQKV")
class Pi0JointQKV(BaseCondition):
    """
    Pi0/Pi0.5 Joint QKV alignment strategy.

    Converts VLM backbone output into prefix context consumable by the action expert,
    including dimension projection and attention mask construction.

    Unlike the simple KVCachePrefix, this strategy additionally provides:
      1. prefix/suffix attention mask construction (for joint attention)
      2. Support for paligemma native prefix (no projection needed)
      3. Support for multimodal position_ids tracking
      4. VLM language model reference passing (for joint QKV mode)
    """

    def __init__(self, config):
        super().__init__(config)
        action_cfg = config.framework.get("action_model", {})
        vlm_cfg = config.framework.get("qwenvl", config.framework.get("paligemma", {}))

        self.vlm_hidden_dim = vlm_cfg.get("vl_hidden_dim", vlm_cfg.get("hidden_dim", 2048))
        self.expert_width = action_cfg.get("action_expert_width", 1024)
        self.pi05 = config.framework.get("pi05", True)

        # Detect VLM type
        self.vlm_type = self._detect_vlm_type(config)

        # Dimension alignment projection (VLM dim -> expert dim)
        # PaliGemma's encode_prefix already outputs expert_width, no projection needed
        if self.vlm_type != "paligemma" and self.vlm_hidden_dim != self.expert_width:
            self.prefix_proj = nn.Linear(
                self.vlm_hidden_dim, self.expert_width, bias=False
            )
        else:
            self.prefix_proj = nn.Identity()

    def _detect_vlm_type(self, config) -> str:
        """Detect VLM type."""
        if hasattr(config.framework, "paligemma") or config.framework.get("paligemma"):
            return "paligemma"
        elif hasattr(config.framework, "qwenvl") or config.framework.get("qwenvl"):
            return "qwen"
        elif hasattr(config.framework, "llamavl") or config.framework.get("llamavl"):
            return "llama"
        return "generic"

    def inject(
        self,
        backbone_output: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert VLM backbone output into prefix context for Pi0 action expert.

        Input:
            backbone_output:
                'features': (B, L, H_vlm) -- VLM final hidden states
                'attention_mask': (B, L) -- padding mask (optional)
                'vlm_language_model': nn.Module -- VLM LM (for joint attention, optional)

        Output:
            Dict:
                'action_context': Dict:
                    'prefix_embs': (B, L, expert_width) -- projected prefix
                    'prefix_pad_masks': (B, L) -- padding mask
                    'prefix_att_masks': (B, L) -- attention type (0=prefix)
                    'vlm_language_model': Optional[nn.Module] -- for joint mode
                    'mode': str -- 'joint_attention' | 'prefix_cache'
                'type': 'kv_cache'
        """
        features = backbone_output["features"]  # (B, L, H_vlm)
        B, L, _ = features.shape
        device = features.device

        # Project to expert width
        prefix_embs = self.prefix_proj(features)  # (B, L, expert_width)

        # Padding mask
        if "attention_mask" in backbone_output and backbone_output["attention_mask"] is not None:
            prefix_pad_masks = backbone_output["attention_mask"].bool()
        else:
            prefix_pad_masks = torch.ones(B, L, dtype=torch.bool, device=device)

        # Attention type mask: prefix tokens marked as 0 (everyone can attend to them)
        prefix_att_masks = torch.zeros(B, L, dtype=prefix_embs.dtype, device=device)

        # Determine running mode
        vlm_lm = backbone_output.get("vlm_language_model", None)
        if vlm_lm is not None and self.vlm_type == "paligemma":
            mode = "joint_attention"
        else:
            mode = "prefix_cache"

        return {
            "action_context": {
                "prefix_embs": prefix_embs,
                "prefix_pad_masks": prefix_pad_masks,
                "prefix_att_masks": prefix_att_masks,
                "vlm_language_model": vlm_lm,
                "mode": mode,
            },
            "type": "kv_cache",
        }

    def get_action_head_input_spec(self) -> Dict[str, Any]:
        """get action head input spec"""
        return {
            "type": "kv_cache",
            "hidden_dim": self.expert_width,
            "pi05": self.pi05,
            "vlm_type": self.vlm_type,
        }
