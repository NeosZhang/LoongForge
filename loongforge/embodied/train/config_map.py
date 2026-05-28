# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""model-name → YAML config path mapping."""

from pathlib import Path

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "models" / "embodied"

MODEL_CONFIG_MAP = {
    # Pi0.5 series
    "pi05": "pi05.yaml",
    # GR00T series
    "groot_n1_6": "groot_n1_6.yaml",
    "xvla": "xvla.yaml",
    "qwen_fast": "qwen_fast.yaml",
}


def get_config_path(model_name: str) -> str:
    """Look up config YAML path by model name."""
    key = model_name.lower().replace("-", "_")
    if key not in MODEL_CONFIG_MAP:
        available = ", ".join(sorted(MODEL_CONFIG_MAP.keys()))
        raise ValueError(f"Unknown model-name '{model_name}'. Available: [{available}]")
    path = _CONFIGS_DIR / MODEL_CONFIG_MAP[key]
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return str(path)
