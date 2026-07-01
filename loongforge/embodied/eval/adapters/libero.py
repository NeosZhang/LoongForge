# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO benchmark adapter.

This adapter is simulator-side and framework-agnostic. It converts between
LIBERO-native observations/actions and the Canonical protocol used by
`vla_eval`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.adapters.base import BaseBenchmarkAdapter

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert LIBERO/robosuite quaternion `[x, y, z, w]` to axis-angle."""
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    """Map model open-gripper scalar to LIBERO gripper command.

    LIBERO eval convention:
    - model value > 0.5 means open
    - LIBERO action gripper is -1 for open, +1 for close
    """
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    value = float(arr[0])
    return np.asarray([1.0 - 2.0 * (value > 0.5)], dtype=np.float32)


class LiberoAdapter(BaseBenchmarkAdapter):
    """Provide LiberoAdapter behavior."""

    def __init__(
        self,
        suite_name: str = "libero_goal",
        robot_setup: str = "franka",
        control_hz: int = 20,
        episodes_per_task: int = 50,
        resolution: int = LIBERO_ENV_RESOLUTION,
        continuous_gripper: bool = False,
    ) -> None:
        """Run __init__."""
        if suite_name not in SUITE_MAX_STEPS:
            raise ValueError(f"Unknown LIBERO suite {suite_name!r}; choose one of {sorted(SUITE_MAX_STEPS)}")
        self.suite_name = suite_name
        self.robot_setup = robot_setup
        self.control_hz = control_hz
        self.episodes_per_task = episodes_per_task
        self.resolution = resolution
        self.continuous_gripper = continuous_gripper

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run obs_to_canonical."""
        primary = np.ascontiguousarray(env_obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(env_obs["robot0_eye_in_hand_image"][::-1, ::-1])

        eef_pos = np.asarray(env_obs.get("robot0_eef_pos"), dtype=np.float32).reshape(-1)[:3]
        eef_rot = quat_to_axisangle(np.asarray(env_obs.get("robot0_eef_quat"), dtype=np.float32))
        gripper_qpos = np.asarray(env_obs.get("robot0_gripper_qpos"), dtype=np.float32).reshape(-1)
        gripper = float(gripper_qpos[0]) if gripper_qpos.size else None

        return {
            "instruction": str(context["instruction"]),
            "images": {
                "primary": primary,
                "wrist": wrist,
                "left": None,
                "right": None,
                "head": None,
            },
            "state": {
                "eef_pos": eef_pos.tolist(),
                "eef_rot_axis_angle": eef_rot.tolist(),
                "gripper": gripper,
                "joint": None,
                "frame": "base",
                "units": {"pos": "m", "rot": "rad"},
            },
            "meta": {
                "benchmark": "libero",
                "robot_setup": self.robot_setup,
                "control_hz": self.control_hz,
                "episode_id": str(context.get("episode_id", "default")),
                "episode_step": int(context.get("episode_step", 0)),
                "runtime": "sim",
                "bimanual": False,
                "chunk_policy": "full",
                "realtime_deadline_ms": None,
            },
        }

    def action_from_canonical(self, canonical_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Run action_from_canonical."""
        if "world_vector" in canonical_action:
            world_vector = np.asarray(canonical_action["world_vector"], dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(canonical_action["rotation_delta"], dtype=np.float32).reshape(-1)
            gripper_value = canonical_action["gripper"]
        elif "actions" in canonical_action:
            flat = np.asarray(canonical_action["actions"], dtype=np.float32).reshape(-1)
            world_vector = flat[:3]
            rotation_delta = flat[3:6]
            gripper_value = flat[6]
        else:
            raise ValueError("Canonical action must contain either structured fields or `actions` flat array")

        if world_vector.size != 3 or rotation_delta.size != 3:
            raise ValueError(f"Invalid LIBERO action shape: world={world_vector.shape}, rot={rotation_delta.shape}")

        gripper = (
            np.asarray([float(gripper_value)], dtype=np.float32)
            if self.continuous_gripper
            else binarize_gripper_open(gripper_value)
        )
        return np.concatenate([world_vector, rotation_delta, gripper], axis=0).astype(np.float32).tolist()

    def get_eval_context(self) -> Dict[str, Any]:
        """Run get_eval_context."""
        return {
            "benchmark": "libero",
            "robot_setup": self.robot_setup,
            "control_hz": self.control_hz,
            "max_steps": {self.suite_name: SUITE_MAX_STEPS[self.suite_name]},
            "action_scale": {
                "pos_scale": 1.0,
                "rot_scale": 1.0,
                "gripper_scale": 1.0,
                "gripper_bias": 0.0,
                "left": None,
                "right": None,
            },
            "bimanual": False,
            "has_state_fields": ["eef_pos", "eef_rot_axis_angle", "gripper"],
            "episodes_per_task": self.episodes_per_task,
            "runtime": "sim",
            "success_oracle_type": "info_flag",
            "suite_name": self.suite_name,
            "num_steps_wait": 10,
            "dummy_action": LIBERO_DUMMY_ACTION,
            "resolution": self.resolution,
        }
