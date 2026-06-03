# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Registry base class — general-purpose name-to-class mapping with decorator registration."""

from typing import Callable, Dict, Type


class Registry:
    """General registry, supports decorator registration and lookup by name."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str) -> Callable:
        """Decorator: register a class to this registry.

        Usage:
            @MY_REGISTRY.register("Foo")
            class Foo:
                ...
        """
        def decorator(cls):
            if key in self._registry:
                raise ValueError(
                    f"[{self._name}] Key '{key}' already registered "
                    f"by {self._registry[key].__name__}, "
                    f"cannot re-register with {cls.__name__}"
                )
            self._registry[key] = cls
            return cls
        return decorator

    def __getitem__(self, key: str) -> Type:
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"[{self._name}] '{key}' not found. Available: [{available}]"
            )
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self):
        """Return all registered keys."""
        return self._registry.keys()

    def items(self):
        """Return all registered (key, class) pairs."""
        return self._registry.items()

    def __repr__(self):
        return f"Registry(name={self._name}, entries={list(self._registry.keys())})"

"""Model-layer registry instances (architecture, condition, action)."""

# Network structure
ARCHITECTURE_REGISTRY = Registry("architectures")

# Modality alignment / condition injection
CONDITION_REGISTRY = Registry("conditions")

# Action strategy
ACTION_REGISTRY = Registry("actions")
