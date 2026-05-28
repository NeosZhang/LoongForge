"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

import importlib
import importlib.metadata


def is_package_available(pkg_name: str, import_name: str | None = None) -> bool:
    """Check if a package is available."""
    if import_name is None:
        import_name = pkg_name
    return importlib.util.find_spec(import_name) is not None


# ML package availability
_transformers_available = is_package_available("transformers")