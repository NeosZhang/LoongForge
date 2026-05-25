"""
Registry - Registration center for each layer's implementations

Uses Decorator pattern to register concrete implementations for each layer, Builder looks up by name
"""

from typing import Dict, Any, Callable, Type


class Registry:
    """General registry, supports decorator registration and lookup by name."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str) -> Callable:
        """Decorator: register a class to this registry.

        Usage:
            @CONDITION_REGISTRY.register("GlobalProjection")
            class GlobalProjection(BaseCondition):
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
        """keys"""
        return self._registry.keys()

    def items(self):
        """items"""
        return self._registry.items()

    def __repr__(self):
        return f"Registry(name={self._name}, entries={list(self._registry.keys())})"


# ═══════════════════════════════════════════════════════════════
# Independent Registry instances for each of the four layers
# ═══════════════════════════════════════════════════════════════

# Layer 1: Training paradigm
TRAINER_REGISTRY = Registry("trainers")

# Layer 2: Network structure
ARCHITECTURE_REGISTRY = Registry("architectures")

# Layer 3: Modality alignment
CONDITION_REGISTRY = Registry("conditions")

# Layer 4: Action
ACTION_REGISTRY = Registry("actions")
