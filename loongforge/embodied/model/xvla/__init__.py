"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

from loongforge.embodied.model.xvla.configuration_xvla import XVLAConfig
from loongforge.embodied.model.xvla.processor_xvla import (
    XVLAAddDomainIdProcessorStep,
    XVLAImageNetNormalizeProcessorStep,
    XVLAImageToFloatProcessorStep,
)

__all__ = [
    "XVLAConfig",
    "XVLAAddDomainIdProcessorStep",
    "XVLAImageNetNormalizeProcessorStep",
    "XVLAImageToFloatProcessorStep",
]
