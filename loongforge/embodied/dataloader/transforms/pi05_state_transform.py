"""
Pi05StateTransform - State discretization transform for PI0.5

Discretizes normalized state into 256 bins and embeds it into the language prompt.
Used when framework.backbone.discrete_state_input is true.

Input data format:
    {"image": [...], "lang": str, "action": ndarray, "state": ndarray}

Output data format:
    {"image": [...], "lang": "Task: <task>, State: <bins>;\nAction: ", "action": ndarray, "state": ndarray}
"""

from typing import Any, Dict, List

import numpy as np

from dataloader.transforms.base import BaseTransform


class Pi05StateTransform(BaseTransform):
    """Discretize state and embed into language prompt (pi0.5 style).

    Converts:
        state (normalized [-1,1]) -> 256-bin discrete tokens
        lang + state -> "Task: <lang>, State: <d0> <d1> ...;\nAction: "
    """

    def __init__(
        self,
        apply_to: List[str] = None,
        max_state_dim: int = 32,
        num_bins: int = 256,
        training: bool = True,
    ):
        super().__init__(apply_to=apply_to or ["lang"], training=training)
        self.max_state_dim = max_state_dim
        self.num_bins = num_bins

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform to data dictionary."""
        state = data.get("state")
        lang = data.get("lang", "")

        if state is None:
            # No state: simple prompt
            data["lang"] = f"Task: {lang.strip()};\nAction: "
            return data

        # Flatten state to 1D
        state_flat = np.asarray(state, dtype=np.float32).flatten()

        # Pad to max_state_dim
        if len(state_flat) < self.max_state_dim:
            state_flat = np.pad(state_flat, (0, self.max_state_dim - len(state_flat)))
        elif len(state_flat) > self.max_state_dim:
            state_flat = state_flat[:self.max_state_dim]

        # Clip to [-1, 1] and discretize
        state_flat = np.clip(state_flat, -1.0, 1.0)
        bins = np.linspace(-1, 1, self.num_bins + 1)[:-1]
        discretized = np.digitize(state_flat, bins=bins) - 1

        # Build prompt
        cleaned_text = lang.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized))
        data["lang"] = f"Task: {cleaned_text}, State: {state_str};\nAction: "

        return data
