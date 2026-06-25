# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pytest

from loongforge.embodied.eval.adapters.robotwin import ROBOTWIN_ACTION_REORDER, RoboTwinAdapter
from loongforge.embodied.eval.bridges.robotwin_policy import ModelClient
from loongforge.embodied.eval.orchestrator.runners.robotwin_runner import (
    _override_eval_policy_for_smoke,
    _override_step_limit,
    build_command,
    _write_deploy_policy,
)


def fake_robotwin_obs() -> dict:
    """Run fake_robotwin_obs."""
    return {
        "observation": {
            "head_camera": {"rgb": np.full((2, 3, 3), 10, dtype=np.uint8)},
            "left_camera": {"rgb": np.full((2, 3, 3), 20, dtype=np.uint8)},
            "right_camera": {"rgb": np.full((2, 3, 3), 30, dtype=np.uint8)},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }


def test_robotwin_obs_to_canonical_extracts_three_cameras_and_joint_state() -> None:
    """Run test_robotwin_obs_to_canonical_extracts_three_cameras_and_joint_state."""
    adapter = RoboTwinAdapter(task_name="pick_object")

    canonical = adapter.obs_to_canonical(
        fake_robotwin_obs(),
        {"instruction": "pick the object", "episode_id": "robotwin/pick_object/0", "episode_step": 3},
    )

    assert canonical["instruction"] == "pick the object"
    np.testing.assert_array_equal(canonical["images"]["primary"], np.full((2, 3, 3), 10, dtype=np.uint8))
    np.testing.assert_array_equal(canonical["images"]["head"], np.full((2, 3, 3), 10, dtype=np.uint8))
    np.testing.assert_array_equal(canonical["images"]["left"], np.full((2, 3, 3), 20, dtype=np.uint8))
    np.testing.assert_array_equal(canonical["images"]["right"], np.full((2, 3, 3), 30, dtype=np.uint8))
    assert canonical["state"]["joint"] == list(np.arange(14, dtype=np.float32))
    assert canonical["meta"]["benchmark"] == "robotwin"
    assert canonical["meta"]["bimanual"] is True


def test_robotwin_action_from_flat_array_reorders_for_env_step() -> None:
    """Run test_robotwin_action_from_flat_array_reorders_for_env_step."""
    adapter = RoboTwinAdapter()
    raw_action = np.arange(14, dtype=np.float32)

    env_action = adapter.action_from_canonical({"actions": raw_action})

    np.testing.assert_array_equal(env_action, raw_action[ROBOTWIN_ACTION_REORDER])
    assert env_action.dtype == np.float32


def test_robotwin_delta_action_adds_current_joint_before_reorder() -> None:
    """Run test_robotwin_delta_action_adds_current_joint_before_reorder."""
    adapter = RoboTwinAdapter(action_mode="delta")
    current_joint = np.arange(14, dtype=np.float32)
    delta = np.ones(14, dtype=np.float32)

    env_action = adapter.action_from_canonical({"actions": delta}, {"current_joint": current_joint})

    np.testing.assert_array_equal(env_action, (current_joint + delta)[ROBOTWIN_ACTION_REORDER])


def test_robotwin_action_from_bimanual_fields() -> None:
    """Run test_robotwin_action_from_bimanual_fields."""
    adapter = RoboTwinAdapter(reorder_action=False)
    canonical_action = {
        "left": {"world_vector": [0, 1, 2], "rotation_delta": [3, 4, 5], "gripper": 6},
        "right": {"world_vector": [7, 8, 9], "rotation_delta": [10, 11, 12], "gripper": 13},
    }

    env_action = adapter.action_from_canonical(canonical_action)

    np.testing.assert_array_equal(env_action, np.arange(14, dtype=np.float32))


def test_robotwin_rejects_invalid_action_shape() -> None:
    """Run test_robotwin_rejects_invalid_action_shape."""
    adapter = RoboTwinAdapter()

    with pytest.raises(ValueError, match="14D"):
        adapter.action_from_canonical({"actions": [0.0] * 13})


def test_robotwin_eval_context_marks_script_success_oracle() -> None:
    """Run test_robotwin_eval_context_marks_script_success_oracle."""
    context = RoboTwinAdapter(task_name="pick_object", action_mode="abs").get_eval_context()

    assert context["benchmark"] == "robotwin"
    assert context["bimanual"] is True
    assert context["success_oracle_type"] == "script"
    assert context["has_state_fields"] == ["joint"]
    assert context["action_dim"] == 14


def test_robotwin_policy_duplicate_7d_bridge_is_explicit() -> None:
    """Run test_robotwin_policy_duplicate_7d_bridge_is_explicit."""
    client = ModelClient.__new__(ModelClient)
    client.action_bridge = "duplicate_7d"

    action = client._extract_robotwin_action(np.arange(7, dtype=np.float32))

    np.testing.assert_array_equal(
        action, np.concatenate([np.arange(7, dtype=np.float32), np.arange(7, dtype=np.float32)])
    )


def test_robotwin_policy_rejects_7d_without_bridge() -> None:
    """Run test_robotwin_policy_rejects_7d_without_bridge."""
    client = ModelClient.__new__(ModelClient)
    client.action_bridge = "strict_14d"

    with pytest.raises(ValueError, match="14D"):
        client._extract_robotwin_action(np.arange(7, dtype=np.float32))


def test_robotwin_runner_writes_vla_eval_policy_config(tmp_path: pathlib.Path) -> None:
    """Run test_robotwin_runner_writes_vla_eval_policy_config."""
    args = argparse.Namespace(
        policy_ckpt_path="/ckpts/model.pt",
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="loongforge_demo",
        seed=3,
        instruction_type="unseen",
        host="127.0.0.1",
        port=10093,
        unnorm_key="new_embodiment",
        action_mode="abs",
        no_action_reorder=False,
        max_steps=400,
        control_hz=10,
        disable_action_cache=True,
        return_action_chunk=False,
        disable_eval_video_log=True,
        output_dir=str(tmp_path / "run"),
    )
    config_path = tmp_path / "deploy_policy.yml"

    _write_deploy_policy(args, config_path)

    text = config_path.read_text(encoding="utf-8")
    assert "policy_name: loongforge.embodied.eval.bridges.robotwin_policy" in text
    assert "policy_ckpt_path: /ckpts/model.pt" in text
    assert "task_name: adjust_bottle" in text
    assert "host: 127.0.0.1" in text
    assert "port: 10093" in text
    assert "unnorm_key: new_embodiment" in text
    assert "disable_action_cache: true" in text
    assert "eval_video_log: false" in text
    assert "trace_path:" in text
    assert "artifacts/robotwin/adjust_bottle/demo_clean/trace.json" in text


def test_robotwin_policy_writes_trace_file(tmp_path: pathlib.Path) -> None:
    """Run test_robotwin_policy_writes_trace_file."""
    client = ModelClient.__new__(ModelClient)
    client.trace_path = tmp_path / "trace.json"
    client.trace_records = []
    client.episode_id = "robotwin/adjust_bottle/default"
    client.adapter = RoboTwinAdapter(task_name="adjust_bottle", action_mode="abs")

    client._record_trace(
        step=2,
        instruction="adjust bottle",
        joint=np.arange(14, dtype=np.float32),
        raw_action=np.ones(14, dtype=np.float32),
        output_action=np.ones(14, dtype=np.float32) * 2,
        env_action=np.ones(14, dtype=np.float32) * 3,
        response={"data": {"inference_latency_ms": 12.5}},
    )

    payload = json.loads(client.trace_path.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "robotwin"
    assert payload["task_name"] == "adjust_bottle"
    assert payload["steps"][0]["step"] == 2
    assert payload["steps"][0]["raw_action"] == [1.0] * 14
    assert payload["steps"][0]["output_action"] == [2.0] * 14
    assert payload["steps"][0]["env_action"] == [3.0] * 14


def test_robotwin_runner_builds_eval_policy_command() -> None:
    """Run test_robotwin_runner_builds_eval_policy_command."""
    args = argparse.Namespace(
        robotwin_python="/envs/robotwin/bin/python",
        robotwin_path="/workspace/RoboTwin",
        policy_ckpt_path="/ckpts/model.pt",
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="loongforge_demo",
        seed=3,
    )

    command = build_command(args, pathlib.Path("/tmp/deploy_policy.yml"))

    assert command[:3] == ["/envs/robotwin/bin/python", "/workspace/RoboTwin/script/eval_policy.py", "--config"]
    assert "--policy_ckpt_path" not in command
    assert "loongforge.embodied.eval.bridges.robotwin_policy" in command
    assert "adjust_bottle" in command


def test_robotwin_step_limit_override_round_trip(tmp_path: pathlib.Path) -> None:
    """Run test_robotwin_step_limit_override_round_trip."""
    robotwin_path = tmp_path / "RoboTwin"
    config_dir = robotwin_path / "task_config"
    config_dir.mkdir(parents=True)
    step_limit_path = config_dir / "_eval_step_limit.yml"
    step_limit_path.write_text("adjust_bottle: 400\n", encoding="utf-8")

    original_text = _override_step_limit(robotwin_path, "adjust_bottle", 5)

    assert original_text == "adjust_bottle: 400\n"
    assert "adjust_bottle: 5" in step_limit_path.read_text(encoding="utf-8")
    step_limit_path.write_text(original_text, encoding="utf-8")
    assert step_limit_path.read_text(encoding="utf-8") == "adjust_bottle: 400\n"


def test_robotwin_eval_policy_override_round_trip(tmp_path: pathlib.Path) -> None:
    """Run test_robotwin_eval_policy_override_round_trip."""
    robotwin_path = tmp_path / "RoboTwin"
    script_dir = robotwin_path / "script"
    script_dir.mkdir(parents=True)
    eval_policy_path = script_dir / "eval_policy.py"
    eval_policy_path.write_text(
        "    st_seed = 100000 * (1 + seed)\n    test_num = 100\n    expert_check = True\n",
        encoding="utf-8",
    )

    original_text = _override_eval_policy_for_smoke(robotwin_path, 1, True, 100001)

    text = eval_policy_path.read_text(encoding="utf-8")
    assert original_text == "    st_seed = 100000 * (1 + seed)\n    test_num = 100\n    expert_check = True\n"
    assert "    st_seed = 100001\n" in text
    assert "    test_num = 1\n" in text
    assert "    expert_check = False\n" in text
    eval_policy_path.write_text(original_text, encoding="utf-8")
    assert (
        eval_policy_path.read_text(encoding="utf-8")
        == "    st_seed = 100000 * (1 + seed)\n    test_num = 100\n    expert_check = True\n"
    )
