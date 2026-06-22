# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""model-name → (YAML path, ModelConfig class, DataConfig class) routing.

MODEL_SCHEMA is the single place that binds a model name to:
  - its shared YAML file (with ``model:`` / ``data:`` sections), and
  - the two typed config classes that the YAML sections are merged into.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from loongforge.embodied.model.pi05.model_configuration_pi05 import Pi05ModelConfig
from loongforge.embodied.data.transforms.pi05.data_configuration_pi05 import Pi05DataConfig
from loongforge.embodied.model.groot_n1_6.model_configuration_groot_n1_6 import GrootN1d6ModelConfig
from loongforge.embodied.data.transforms.groot_n1_6.data_configuration_groot_n1_6 import GrootN1d6DataConfig
from loongforge.embodied.model.xvla.model_configuration_xvla import XvlaModelConfig
from loongforge.embodied.data.transforms.xvla.data_configuration_xvla import XvlaDataConfig
from loongforge.embodied.model.fastwam.modeling_configuration_fastwam import FastWAMModelConfig
from loongforge.embodied.data.transforms.fastwam.data_configuration_fastwam import FastWAMDataConfig

_CONFIGS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "configs" / "models" / "embodied"
)


@dataclass(frozen=True)
class ModelSchema:
    """Binding of a model name to its YAML file and typed config classes."""

    yaml_file: str
    model_config_cls: Type
    data_config_cls: Type


MODEL_SCHEMA = {
    "pi05": ModelSchema("pi05.yaml", Pi05ModelConfig, Pi05DataConfig),
    "groot_n1_6": ModelSchema("groot_n1_6.yaml", GrootN1d6ModelConfig, GrootN1d6DataConfig),
    "xvla": ModelSchema("xvla.yaml", XvlaModelConfig, XvlaDataConfig),
    "fastwam": ModelSchema("fastwam.yaml", FastWAMModelConfig, FastWAMDataConfig),
}


def _normalize(model_name: str) -> str:
    return model_name.lower().replace("-", "_")


def get_model_schema(model_name: str) -> ModelSchema:
    """Look up the ModelSchema for a model name."""
    key = _normalize(model_name)
    if key not in MODEL_SCHEMA:
        available = ", ".join(sorted(MODEL_SCHEMA.keys()))
        raise ValueError(f"Unknown model-name '{model_name}'. Available: [{available}]")
    return MODEL_SCHEMA[key]


def get_config_path(model_name: str) -> str:
    """Look up config YAML path by model name."""
    schema = get_model_schema(model_name)
    path = _CONFIGS_DIR / schema.yaml_file
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return str(path)
