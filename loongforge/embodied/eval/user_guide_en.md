# LoongForge-VLA Offline Evaluation — User Guide

LoongForge-VLA offline evaluation runs a policy against a simulation benchmark from a single YAML config. You point the CLI at one `--config <yaml>` file; it launches the benchmark simulator and an independent model server, runs the rollout, and writes results.

Two policies are supported out of the box — **pi05** and **xvla** — across five benchmarks: **LIBERO, CALVIN, SimplerEnv, RoboTwin, ManiSkill**.

---

## 1. What's supported

### Models

| Model | Notes |
|---|---|
| **pi05** | π0.5 flow-matching policy |
| **xvla** | X-VLA multi-embodiment policy (uses a per-benchmark `domain_id`) |

Any model that implements the shared `predict_action(images, instructions, state=None, dataset_stats=None)` interface can be added — see §7.

### Benchmarks and released-weight coverage

The matrix below shows which model × benchmark combinations have released weights with verified task success, versus those that run today only as a connectivity check (random-initialized weights, useful to validate the pipeline but not a score).

| | LIBERO | CALVIN | SimplerEnv (WidowX) | RoboTwin | ManiSkill |
|---|---|---|---|---|---|
| **pi05** | ✅ task success | connectivity only¹ | connectivity only² | ✅ task success | connectivity only³ |
| **xvla** | ✅ task success | connectivity only⁴ | ✅ task success⁵ | ✅ task success | connectivity only³ |

¹ No CALVIN-domain pi05 weights released yet.
² No Bridge/WidowX fine-tuned pi05 weights yet.
³ No ManiSkill-domain weights released yet.
⁴ Needs official X-VLA CALVIN (ABC_D) weights.
⁵ Requires the SimplerEnv absolute-EE control patch (see §4.3).

"Task success" means at least one episode passed the benchmark's official success criterion.

### Features

- Single-YAML entry point; the CLI only takes `--config`.
- Benchmark client and model server run as separate processes and communicate over WebSocket + msgpack RPC, so simulator and model dependencies stay isolated.
- Per-episode results, suite aggregates, optional replay GIFs and per-step action traces.
- Model-agnostic: benchmark code never imports model code, and vice versa.

---

## 2. Quick start

### 2.1 Prepare environments

Use two separate conda environments (they usually have conflicting dependencies):

- **Benchmark environment** — the simulator client, one per benchmark (e.g. a `libero` env, a `simplerenv` env, …).
- **Model server environment** — loads pi05/xvla; e.g. a `loongforge` env. Requires Python ≥ 3.12.

Per-benchmark dependency details are in [benchmark_envs.md](benchmark_envs.md).

### 2.2 Get the weights

Released weights (Hugging Face):

| Combo | Weights |
|---|---|
| xvla + LIBERO | [2toINF/X-VLA-LIBERO](https://huggingface.co/2toINF/X-VLA-LIBERO) |
| xvla + RoboTwin | [2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2) |
| xvla + SimplerEnv | [2toINF/X-VLA-WidowX](https://huggingface.co/2toINF/X-VLA-WidowX) |
| pi05 + LIBERO | a π0.5 LIBERO fine-tune (`model.safetensors` + `dataset_statistics.json`) |
| pi05 + RoboTwin | a π0.5 RoboTwin-2.0 joint fine-tune (`model.safetensors` + openpi `norm_stats`, see §4.4) |

### 2.3 Configure and run

Example configs live under `examples/embodied/<model>/eval/`. Each benchmark ships a ready-to-edit template:

```bash
cd /path/to/LoongForge-VLA

# 1. Copy/edit the template config: fill in weight paths and the two python envs.
#    examples/embodied/pi05/eval/configs/libero/object_smoke.yaml

# 2. Run it.
examples/embodied/pi05/eval/run_libero_eval.sh
```

The run script accepts a few environment overrides: `CONFIG`, `REPO_ROOT`, `BENCHMARK_PYTHON`, `CUDA_VISIBLE_DEVICES`, and (for SAPIEN benchmarks) `LD_LIBRARY_PATH` / `VK_ICD_FILENAMES`.

Or invoke the orchestrator directly:

```bash
python -m loongforge.embodied.eval.orchestrator.run --config /path/to/config.yaml
```

Run the orchestrator command inside the **benchmark** conda environment; the config's `server.python` points at the **model server** environment.

---

## 3. Configuration reference

A config has five sections: `benchmark`, `model`, `server`, `run`, `timeouts`.

```yaml
benchmark:
  name: libero               # libero | calvin | simplerenv | robotwin | maniskill
  suite: libero_object       # benchmark-specific
  max_tasks: 1
  episodes_per_task: 1
  max_steps: 300
  num_steps_wait: 10

model:
  backend: loongforge        # loongforge | mock
  model_type: pi05           # REQUIRED (no default) — pi05 | xvla
  action_dim: 7
  action_horizon: 50
  # Optional model capability fields (have defaults; usually omitted):
  #   state_encoding   proprio encoding the model consumes
  #   action_encoding  the model's action encoding
  #   domain_id        xvla multi-embodiment id (auto by benchmark if omitted)

server:
  host: 127.0.0.1
  port: 12093
  health_port: 12094
  python: /path/to/model-server-env/bin/python
  log: /path/to/policy_server.log
  start_timeout_sec: 900
  ckpt_path: /path/to/checkpoint_or_model_dir
  dataset_statistics_path: /path/to/dataset_statistics.json
  tokenizer_path: /path/to/paligemma-3b-pt-224
  use_bf16: false
  loongforge_root: /path/to/LoongForge-VLA
  random_init: false         # true = run with random weights (connectivity check)

run:
  output_dir: /path/to/reports/pi05/libero/object_smoke
  seed: 7
  save_trace: true
  save_replay: true

timeouts:
  policy_call_ms: 600000
  per_step_sec: 600
  per_episode_sec: 900
```

Key fields:

- `benchmark.name` — selects the benchmark runner.
- `model.model_type` — **required**; selects the model factory / PayloadBuilder (`pi05` | `xvla`). There is no default — the eval server fails fast if it is missing.
- `model.backend` — `loongforge` for a real model, `mock` for a pipeline-only check.
- `model:` — model-structure fields (`action_dim`, `action_horizon`, …) plus optional capability fields (`state_encoding` / `action_encoding` / `domain_id`). Defaults are sensible per model, so you rarely set these by hand.
- `server.ckpt_path` — a directory with `model.safetensors` (or the weight file). Set `server.random_init: true` to run without weights.
- `server.dataset_statistics_path` — action-normalization stats the model uses internally (e.g. pi05).
- `server.python` — the model server interpreter.
- `run.output_dir` — where results are written.
- Under disk pressure, set `run.save_replay: false` and `run.save_trace: false` to keep only the CSV/JSONL summaries.

Every config comes as a **public template** (with `/path/to/...` placeholders, meant to be edited) plus a matching launch script. Optional knobs (larger suites, more episodes) are documented in each config's header comments — raise `max_tasks` / `episodes_per_task` in the same file.

---

## 4. Running each benchmark

### 4.1 LIBERO

```bash
examples/embodied/pi05/eval/run_libero_eval.sh   # pi05
examples/embodied/xvla/eval/run_libero_eval.sh   # xvla
```

- pi05: `action_dim: 7`, `action_horizon: 50`, plus a matching `dataset_statistics.json`.
- xvla: set `model.domain_id: 3` (or omit to auto-resolve); recommended `max_steps: 800`.

Change the suite (`libero_object` / `libero_spatial` / `libero_goal` / `libero_10`) and episode counts in the config.

### 4.2 CALVIN

CALVIN is long-horizon Franka language manipulation (five subtasks per sequence). `benchmark.dataset_path` must point at a tree containing `validation/`.

No CALVIN-domain weights are released yet, so the shipped configs run with `server.random_init: true` (connectivity check only). With a CALVIN-domain checkpoint, set `random_init: false`, fill `ckpt_path` / `dataset_statistics_path`, and for xvla set `model.domain_id: 2`.

### 4.3 SimplerEnv

```bash
examples/embodied/xvla/eval/run_simplerenv_eval.sh
```

xvla with WidowX-domain weights reaches task success. Two notes:

- Set `benchmark.control_mode: arm_pd_ee_target_base_pose_gripper_pd_joint_pos`, `max_steps: 1200`, and `model.domain_id: 0` (or omit to auto-resolve).
- Upstream SimplerEnv does not ship absolute end-effector control by default. Apply the patch described in `examples/embodied/xvla/eval/SIMPLERENV_PATCH_en.md` first.

pi05 has no Bridge/WidowX weights yet, so its SimplerEnv configs are connectivity checks (`random_init: true`).

### 4.4 RoboTwin

RoboTwin reuses its official evaluator, which calls the policy through a plugin. Select the protocol with `benchmark.action_bridge`:

| `action_bridge` | Model | Notes |
|---|---|---|
| `pi05_aloha_14d` | pi05 | openpi Aloha joint protocol; `action_dim: 14`, `action_horizon: 32` |
| `ee6d_dual` | xvla | X-VLA dual-arm end-effector protocol; `domain_id: 6` |

```bash
examples/embodied/pi05/eval/run_robotwin_eval.sh    # pi05, action_bridge: pi05_aloha_14d
examples/embodied/xvla/eval/run_robotwin_eval.sh    # xvla, action_bridge: ee6d_dual
```

For **pi05 + RoboTwin**, the model needs a `dataset_statistics.json` derived from the checkpoint's openpi `norm_stats.json`. A ready-made copy ships at `examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json`; to regenerate from another openpi-style checkpoint:

```python
import json
from pathlib import Path

raw = json.loads(Path("<ckpt>/assets/.../norm_stats.json").read_text())["norm_stats"]
out = {"observation.state": raw["state"], "action": raw["actions"]}
Path("dataset_stats.json").write_text(json.dumps(out, indent=2))
```

### 4.5 ManiSkill

```bash
examples/embodied/pi05/eval/run_maniskill_eval.sh
```

Default task `PickCube-v1` with `pd_ee_delta_pose` control. No ManiSkill-domain weights are released yet, so shipped configs run with `random_init: true`. With a ManiSkill-domain checkpoint, set `random_init: false` and provide matching stats (state dim is typically 8: 7 arm joints + gripper width).

---

## 5. Outputs

LIBERO / CALVIN / SimplerEnv / ManiSkill write under `run.output_dir`:

| File | Meaning |
|---|---|
| `results.jsonl` | per-episode results |
| `summary.csv` | task-level aggregate |
| `suite_summary.csv` | suite-level aggregate |
| `artifacts/.../replay_*.gif` | replay (when `save_replay: true`) |
| `artifacts/.../trace_*.json` | per-step action trace (when `save_trace: true`) |
| `policy_server.log` | model server stdout/stderr |

RoboTwin additionally collects the official evaluator logs, deploy config, result file, and any `mp4` videos under `artifacts/robotwin/<task_name>/<task_config>/`, and writes one `results.jsonl` row per completed episode so aggregation matches the other benchmarks.

`run.output_dir` is a stable run tag; by default the orchestrator writes a timestamped subdirectory so previous results are never overwritten. Set `run.timestamped_output: false` to reuse a fixed directory.

---

## 6. Troubleshooting

**Vulkan / SAPIEN (SimplerEnv, RoboTwin, ManiSkill).** These render with SAPIEN and need a working NVIDIA Vulkan ICD. Check with `vulkaninfo` (not just `nvidia-smi`):

```bash
LD_LIBRARY_PATH=/path/to/nvidia_lib:/usr/lib64 \
VK_ICD_FILENAMES=/path/to/nvidia_icd.json \
vulkaninfo
```

Expect `deviceName = NVIDIA ...` / `driverName = NVIDIA`. If you see only `llvmpipe` / `lavapipe`, camera images and replays are unreliable. Set `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR` **before** SAPIEN is imported; the runners re-exec the process so these take effect.

**MuJoCo (LIBERO, CALVIN).** Keep the `MUJOCO_GL` / `PYOPENGL_PLATFORM` and benchmark config-path settings from the shipped configs.

**Disk space.** Replay GIFs are the largest artifact. Set `run.save_replay: false` (and `save_trace: false`) if a run fails while writing artifacts.

**Python version.** The model server needs Python ≥ 3.12 (`lerobot==0.5.0` requirement); don't run it in an old 3.10 env.

---

## 7. Adding a new model

Beyond pi05/xvla, a new model plugs into three small pieces (benchmark runners and adapters are reused unchanged):

1. **Model factory** (`factories/<model>_factory.py`): loads the model + checkpoint and returns it behind the shared `predict_action(images, instructions, state=None, dataset_stats=None)` interface.
2. **PayloadBuilder** (`payload_builders/<model>.py`): turns a benchmark observation into the model's `predict_action` inputs, and declares the model's capabilities (`state_encoding` / `action_encoding` / `domain_id`).
3. **ActionDecoder** (`action_decoders/`, optional): converts the model's raw actions into the benchmark's action space. If the model's action encoding already matches the benchmark, this is a no-op and nothing is needed.

Action normalization/unnormalization always lives inside the model's `predict_action`; eval only passes stats through. The full step-by-step checklist (with pi05 and xvla as worked examples) and the exact interface contract are in [model_integration.md](model_integration.md).

---

## 8. Related docs

| Doc | Contents |
|---|---|
| [README.md](README.md) | Module scope and quick start |
| [benchmark_envs.md](benchmark_envs.md) | Per-benchmark conda environments and dependencies |
| [model_integration.md](model_integration.md) | Step-by-step guide to add a model + the `predict_action` contract for model owners |
| `examples/embodied/xvla/eval/SIMPLERENV_PATCH_en.md` | SimplerEnv absolute end-effector control patch |

---

## Appendix: how a step flows (optional)

For readers who want the internals, one evaluation step is a four-stage chain that keeps the model and benchmark decoupled:

```
Adapter            benchmark observation      → canonical observation
PayloadBuilder     canonical observation      → model.predict_action(**kwargs)
model.predict_action (server)                 → raw action chunk
ActionDecoder      raw action chunk           → benchmark env action
```

- **Adapter** (one per benchmark) reads the raw observation and exposes images, language, and raw robot state.
- **PayloadBuilder** (one per model) assembles the model's inputs and encodes proprio per the model's `state_encoding`.
- The **model server** runs `predict_action` and owns all normalization.
- **ActionDecoder** converts the model's output into the env's action space; the right decoder is selected automatically from the model's `action_encoding` and the benchmark's action space.
