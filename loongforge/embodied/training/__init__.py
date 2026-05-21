"""
LoongForgeVLA Training - Training paradigms and general training infrastructure

Provides:
  - Unified training entry point (train_layered.py)
  - Trainer hierarchy (trainers/)
  - General training utilities (trainer_utils/)
    - overwatch: Distributed-aware logging
    - trainer_tools: Parameter freezing, LR groups, checkpoint management
    - peft: LoRA configuration, injection, checkpoint saving/merging
"""
