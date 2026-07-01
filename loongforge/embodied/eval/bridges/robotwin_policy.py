# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin official-evaluator policy bridge for the standalone eval module.

RoboTwin is special because the eval module reuses RoboTwin's official
evaluator, which imports a policy plugin by `policy_name`; LIBERO and
SimplerEnv are controlled directly by local runners.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.adapters.robotwin import ROBOTWIN_ACTION_DIM, ROBOTWIN_DEFAULT_MAX_STEPS, RoboTwinAdapter


class ModelClient:
    """Provide ModelClient behavior."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10093,
        unnorm_key: Optional[str] = None,
        task_name: str = "robotwin_task",
        robot_setup: str = "bimanual_dual_arm",
        control_hz: int = 10,
        max_steps: int = ROBOTWIN_DEFAULT_MAX_STEPS,
        action_mode: str = "abs",
        reorder_action: bool = True,
        timeout: float = 300,
        disable_action_cache: bool = False,
        return_action_chunk: bool = False,
        action_bridge: str = "strict_14d",
        trace_path: Optional[str] = None,
    ) -> None:
        """Run __init__."""
        from loongforge.embodied.eval.transport import PolicyClient

        self.client = PolicyClient(host=host, port=port, timeout=timeout)
        self.adapter = RoboTwinAdapter(
            task_name=task_name,
            robot_setup=robot_setup,
            control_hz=control_hz,
            max_steps=max_steps,
            action_mode=action_mode,
            reorder_action=reorder_action,
        )
        self.unnorm_key = unnorm_key
        self.disable_action_cache = disable_action_cache
        self.return_action_chunk = return_action_chunk
        self.action_bridge = action_bridge
        self.task_description: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.prev_action: Optional[np.ndarray] = None
        self.initial_joint: Optional[np.ndarray] = None
        self.trace_path = pathlib.Path(trace_path) if trace_path else None
        self.trace_records: List[Dict[str, Any]] = []

    def reset(self, task_description: str = "", episode_id: Optional[str] = None) -> None:
        """Run reset."""
        self.task_description = task_description
        self.episode_id = episode_id or f"robotwin/{self.adapter.task_name}/{task_description or 'default'}"
        self.prev_action = None
        self.initial_joint = None
        self.trace_records = []
        self.client.reset(self.episode_id)
        self._flush_trace()

    def step(self, observation: Dict[str, Any], instruction: str, step: int = 0) -> np.ndarray:
        """Run step."""
        if instruction != self.task_description or self.episode_id is None:
            self.reset(task_description=instruction)

        joint = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        if self.initial_joint is None:
            self.initial_joint = joint.copy()

        canonical_obs = self.adapter.obs_to_canonical(
            observation,
            {
                "instruction": instruction,
                "episode_id": self.episode_id,
                "episode_step": step,
            },
        )
        response = self.client.predict_action(
            images=canonical_obs["images"],
            instruction=canonical_obs["instruction"],
            episode_id=canonical_obs["meta"]["episode_id"],
            episode_step=canonical_obs["meta"]["episode_step"],
            state=canonical_obs["state"],
            meta=canonical_obs["meta"],
            unnorm_key=self.unnorm_key,
            disable_action_cache=self.disable_action_cache,
            return_action_chunk=self.return_action_chunk,
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Policy error: {response}")

        raw_action = self._extract_robotwin_action(response["data"]["actions"])
        action_mode = self.adapter.action_mode
        if action_mode == "delta":
            base = self.prev_action if self.prev_action is not None else joint
            output_action = np.asarray(base, dtype=np.float32).reshape(-1) + raw_action
            self.prev_action = output_action.copy()
            env_action = self.adapter.action_from_canonical({"actions": output_action}, {"action_mode": "abs"})
        elif action_mode == "rel":
            output_action = np.asarray(self.initial_joint, dtype=np.float32).reshape(-1) + raw_action
            env_action = self.adapter.action_from_canonical({"actions": output_action}, {"action_mode": "abs"})
        else:
            output_action = raw_action
            env_action = self.adapter.action_from_canonical({"actions": raw_action})
        self._record_trace(step, instruction, joint, raw_action, output_action, env_action, response)
        return env_action

    def _extract_robotwin_action(self, actions: Any) -> np.ndarray:
        """Run _extract_robotwin_action."""
        flat = np.asarray(actions, dtype=np.float32).reshape(-1)
        if flat.size >= ROBOTWIN_ACTION_DIM:
            return flat[:ROBOTWIN_ACTION_DIM]
        if flat.size == 7 and self.action_bridge == "duplicate_7d":
            return np.concatenate([flat, flat], axis=0).astype(np.float32)
        raise ValueError(
            f"RoboTwin requires {ROBOTWIN_ACTION_DIM}D bimanual actions, got {flat.size}D. "
            "Use a 14D RoboTwin pi05 checkpoint, or set action_bridge='duplicate_7d' only for smoke testing."
        )

    def _record_trace(
        self,
        step: int,
        instruction: str,
        joint: np.ndarray,
        raw_action: np.ndarray,
        output_action: np.ndarray,
        env_action: np.ndarray,
        response: Dict[str, Any],
    ) -> None:
        """Append and persist a RoboTwin step trace record."""
        data = response.get("data", {})
        self.trace_records.append(
            {
                "step": int(step),
                "episode_id": self.episode_id,
                "instruction": instruction,
                "state": np.asarray(joint).tolist(),
                "raw_action": np.asarray(raw_action).tolist(),
                "output_action": np.asarray(output_action).tolist(),
                "env_action": np.asarray(env_action).tolist(),
                "action_mode": self.adapter.action_mode,
                "inference_latency_ms": data.get("inference_latency_ms"),
                "timestamp_sec": time.time(),
            }
        )
        self._flush_trace()

    def _flush_trace(self) -> None:
        """Write trace records when a trace path is configured."""
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "robotwin",
            "task_name": self.adapter.task_name,
            "episode_id": self.episode_id,
            "steps": self.trace_records,
        }
        self.trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        """Run close."""
        self._flush_trace()
        self.client.close()


def get_model(usr_args: Dict[str, Any]) -> ModelClient:
    """Run get_model."""
    return ModelClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 10093)),
        unnorm_key=usr_args.get("unnorm_key"),
        task_name=usr_args.get("task_name") or "robotwin_task",
        robot_setup=usr_args.get("robot_setup", "bimanual_dual_arm"),
        control_hz=int(usr_args.get("control_hz", 10)),
        max_steps=int(usr_args.get("max_steps", ROBOTWIN_DEFAULT_MAX_STEPS)),
        action_mode=usr_args.get("action_mode", "abs"),
        reorder_action=bool(usr_args.get("reorder_action", True)),
        timeout=float(usr_args.get("timeout", 300)),
        disable_action_cache=bool(usr_args.get("disable_action_cache", False)),
        return_action_chunk=bool(usr_args.get("return_action_chunk", False)),
        action_bridge=usr_args.get("action_bridge", "strict_14d"),
        trace_path=usr_args.get("trace_path"),
    )


def reset_model(model: ModelClient) -> None:
    """Run reset_model."""
    model.reset(task_description="")


def eval(TASK_ENV: Any, model: ModelClient, observation: Dict[str, Any]) -> None:
    """Run eval."""
    instruction = str(TASK_ENV.get_instruction())
    action = model.step(observation, instruction=instruction, step=TASK_ENV.take_action_cnt)
    TASK_ENV.take_action(action)
