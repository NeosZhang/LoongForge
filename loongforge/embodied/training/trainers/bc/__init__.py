"""
BC (Behavior Cloning) trainer subpackage

Contains BCBaseTrainer intermediate base class and 6 concrete training stage Trainers.
"""

from training.trainers.bc.bc_base_trainer import BCBaseTrainer
from training.trainers.bc.bc_trainer import BCTrainer

__all__ = [
    "BCBaseTrainer",
    "BCTrainer",
]
