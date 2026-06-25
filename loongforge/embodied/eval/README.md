# LoongForge-VLA Offline Eval

This directory contains the LoongForge-VLA offline evaluation module. It runs benchmark clients and model policy servers as separate processes connected by a WebSocket/msgpack-numpy RPC protocol.

## Scope

- `loongforge.embodied.eval.protocol`: canonical observation/action/result schema.
- `loongforge.embodied.eval.transport`: WebSocket RPC client/server utilities.
- `loongforge.embodied.eval.adapters`: benchmark-side adapters.
- `loongforge.embodied.eval.servers.loongforge_server`: LoongForge policy server entrypoint.
- `loongforge.embodied.eval.servers.loongforge_policy`: LoongForge pi05 adapter.
- `loongforge.embodied.eval.servers.mock_policy`: lightweight protocol mock for tests.

LoongForge source code under `../loongforge` is not patched by this eval module. Model-specific compatibility lives in `eval/servers`.

## Quick Start: pi05 on LIBERO / CALVIN / SimplerEnv / RoboTwin

Use YAML as the only user-facing entrypoint:

```bash
cd /workspace/LoongForge-VLA
examples/embodied/pi05/eval/run_libero_eval.sh
```

The LIBERO simulator runs in the benchmark environment. The policy server is launched from the YAML `server.python` field, typically a Python 3.12 LoongForge environment with `lerobot==0.5.0`.

## Configs

Environment details for each benchmark are recorded in `benchmark_envs.md`. User-editable eval configs and pi05 launch scripts live under `examples/embodied/pi05/eval`. Public scripts use public configs with `/path/to/...` placeholders. Scripts ending in `_internal.sh` call matching `_internal.yaml` configs and are intended to run directly in the internal environment.

- `examples/embodied/pi05/eval/configs/libero/smoke_steps10.yaml`: pi05 LIBERO Goal example.
- `examples/embodied/pi05/eval/configs/libero/spatial_smoke.yaml`: pi05 LIBERO Spatial example.
- `examples/embodied/pi05/eval/configs/libero/object_smoke.yaml`: pi05 LIBERO Object example.
- `examples/embodied/pi05/eval/configs/libero/libero10_smoke.yaml`: pi05 LIBERO 10 example.
- `examples/embodied/pi05/eval/configs/calvin/smoke.yaml`: pi05 CALVIN long-horizon sequence example.
- `examples/embodied/pi05/eval/configs/simplerenv/eggplant_300step.yaml`: pi05 SimplerEnv Bridge eggplant example.
- `examples/embodied/pi05/eval/configs/simplerenv/carrot_on_plate_60step.yaml`: pi05 SimplerEnv Bridge carrot example.
- `examples/embodied/pi05/eval/configs/simplerenv/stack_cube_60step.yaml`: pi05 SimplerEnv Bridge cube stacking example.
- `examples/embodied/pi05/eval/configs/simplerenv/spoon_on_towel_60step.yaml`: pi05 SimplerEnv Bridge spoon example.
- `examples/embodied/pi05/eval/configs/robotwin/random_init_5step.yaml`: RoboTwin official runner check with a random-initialized 14D pi05 model.

The pi05 configs use:

- `model.backend: loongforge`
- `model.model_type: pi05`
- `model.ckpt_path`: checkpoint directory or `model.safetensors`, unless `model.random_init: true` is set for a smoke run
- `model.dataset_statistics_path`: dataset stats used for LoongForge pi05 action unnormalization when using trained weights
- `server.python`: LoongForge Python environment
- `run.output_dir`: runtime output directory under `eval/reports/<model>/<benchmark>/<run_name>/`; `reports/` is generated locally and is not committed

For protocol/debug smoke testing, copy an existing YAML and set `model.backend: mock`; mock backend configs are not shipped as pi05 examples because they are not model-specific.

## Outputs

A benchmark run writes:

- `results.jsonl`: one JSON record per episode
- `summary.csv`: task-level summary
- `suite_summary.csv`: suite-level summary
- `artifacts/.../replay_*.gif`: optional replay for direct rollout runners such as LIBERO and SimplerEnv
- `artifacts/.../trace_*.json` or `artifacts/.../trace.json`: optional per-step action trace
- `artifacts/.../videos/*.mp4`: optional RoboTwin official evaluator videos when video logging is enabled
- `policy_server.log`: model server stdout/stderr

## Verified Runs

The current pi05 integration has been verified end-to-end with internal pi05 checkpoints. Public configs replace checkpoint, tokenizer, dataset statistics, and local driver paths with `/path/to/...` placeholders. Internal launch scripts use matching `_internal.yaml` files so they can run directly in the internal environment.

LIBERO executes 300 environment steps and produces replay/trace artifacts with internal pi05 checkpoints. LeRobot mean/std compatibility is currently a LIBERO weight-adaptation path. CALVIN runs 7D Franka long-horizon sequences and writes `success_count` plus Task 1-5 success rates in `summary.csv`; the internal debug dataset has passed a 30-step smoke with reset, policy RPC, env step, trace, and summary output. Current CALVIN smoke uses q99 unnormalization plus CALVIN adapter scaling with a LIBERO-domain pi05 checkpoint, so it is not a credible long-horizon score. SimplerEnv Bridge runner, Vulkan/SAPIEN headless runtime, reset/step loop, trace output, and replay GIF output have been verified; a fixed small-action sanity rollout confirms the WidowX controller moves. Current internal SimplerEnv pi05 configs use LIBERO-domain weights or no Bridge/WidowX `dataset_statistics_path`, so they are chain smoke/debug runs rather than credible SimplerEnv benchmark scores. For real SimplerEnv evaluation, use a Bridge/WidowX/SimplerEnv-trained pi05 checkpoint and matching dataset statistics. RoboTwin official tasks are bimanual 14D; `random_init_5step.yaml` verifies the official evaluator path with a random-initialized 14D LoongForge pi05 model, writes `artifacts/robotwin/.../trace.json`, collects RoboTwin official `*.mp4` videos when generated, and records 0/1 success as expected for random weights. For real RoboTwin evaluation, use a 14D RoboTwin-trained pi05 checkpoint and matching dataset statistics.

## Development Notes

- Keep model framework logic behind policy server adapters.
- Do not modify `../loongforge` source code for eval-specific compatibility.
- Prefer adding YAML configs over command-line parameter sprawl.
- Use `model.backend` in YAML for backend selection and `benchmark.name` for benchmark selection.
