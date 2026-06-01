# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Layer 3 - BaseCondition: Condition strategy base class

Strategy pattern: defines the transformation interface from backbone features to action head input.
Different strategies convert VLM/WorldModel outputs into representations consumable by the action head.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseCondition(ABC, nn.Module):
    """
    Condition strategy abstract base class.

    Responsibility: convert backbone encoded output (vision/language features) into
    conditioning representations required by the action head.

    Different strategies produce different forms of action_context:
      - GlobalProjection:        Tensor (B, L, H)
      - LayerwiseCrossAttention: List[Tensor], len = N_dit_layers
      - KVCachePrefix:           Dict{"cache": DynamicCache, "mask": Tensor}
      - QFormerFiLM:             Tensor (B, Q, H)
      - LatentFrameInjection:    Tensor (B, C, T, H, W)
      - CrossAttentionFusion:    Tensor (B, N, H)
      - DirectEmbedding:         Tensor (B, N, H)
    """

    def __init__(self, config):
        """
        Args:
            config: OmegaConf configuration (full framework config or condition sub-config)
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def inject(
        self,
        backbone_output: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert backbone encoded results into representations consumable by action head.

        Args:
            backbone_output: Output dict from BaseArchitecture.encode():
                - 'features': Tensor (B, L, H) — backbone final output
                - 'hidden_states': Optional[Tuple[Tensor]] — per-layer hidden states
                - 'auxiliary_outputs': Optional[Dict] — extra outputs (video pred, etc.)

        Returns:
            Dict containing:
                - 'action_context': Main output (type depends on specific strategy)
                - 'type': str — output type identifier ("single_context" | "layerwise_list" |
                                               "kv_cache" | "latent_volume")
                - Other strategy-specific extra outputs
        """
        ...

    @abstractmethod
    def get_action_head_input_spec(self) -> Dict[str, Any]:
        """
        Declare the output specification of this strategy for Builder to validate compatibility with action head.

        Returns:
            Dict containing:
                - 'type': str — "single_context" | "layerwise_list" | "kv_cache" | ...
                - 'hidden_dim': int — output feature dimension
                - 'num_layers': int (layerwise only) — number of layers
        """
        ...

    def extra_repr(self) -> str:
        """add extra info"""
        spec = self.get_action_head_input_spec()
        return f"type={spec.get('type')}, hidden_dim={spec.get('hidden_dim')}"
