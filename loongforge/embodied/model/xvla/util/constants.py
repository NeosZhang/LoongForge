"""
XVLA Module - Extended Vision-Language-Action Model

Core XVLA components including configuration, model, processor, and action spaces.
Provides Florence2 VLM backbone, policy transformer, and various action representations.
"""

# Observation constants
OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_STATE = OBS_STR + ".state"
OBS_IMAGES = OBS_STR + ".images"
OBS_LANGUAGE_TOKENS = OBS_STR + ".language.tokens"

# Action constants
ACTION = "action"

# Policy processor constants
POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

# ImageNet normalization constants
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}