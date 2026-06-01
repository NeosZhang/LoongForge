# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge training"""

try:
    from loongforge.utils import get_args
except ImportError:
    # Allow sub-packages (e.g. loongforge.embodied) to work without Megatron-LM
    import warnings
    warnings.warn(
        "Megatron-LM not found. Only loongforge.embodied (native PyTorch) is available.",
        ImportWarning,
        stacklevel=2,
    )
