---
name: vla-model-eval-adapter
description: Use this skill when adapting a new VLA/model backend into the LoongForge embodied eval system after benchmark runners already exist. Trigger on requests to add a new model to eval, create model policy adapters, write eval YAML, wire model.backend/server routing, validate action normalization or dataset_statistics, or reproduce pi05-style model integration. By default, generate demo YAMLs and smoke-test plans for every already-supported benchmark, such as LIBERO, CALVIN, SimplerEnv, RoboTwin, and ManiSkill, not just one benchmark. This skill should be used even when the user only mentions YAML and adapter work, because the model-side policy is the architecture adapter.
---

# VLA Model Eval Adapter

Use this skill to connect a new model backend to the existing LoongForge embodied evaluation stack. The benchmark runners and benchmark adapters are assumed to already exist. Your job is to add or adapt the model side so the model can be launched by YAML, receive the unified policy RPC payload, return benchmark-compatible actions, and produce smoke-test evidence.

## Mental model

Keep the two adapter layers separate:

```text
benchmark adapter:
  benchmark obs/action <-> canonical eval schema
  examples: adapters/maniskill.py, adapters/simplerenv.py

model policy adapter:
  canonical eval payload <-> concrete model inference API
  examples: servers/loongforge_policy.py, servers/<model>_policy.py
```

In this architecture, `xxxxx_policy.py` is the model-side adapter even if the class is named `Policy`. It translates the unified eval RPC payload into the model's real inference API and translates model outputs back into the unified `actions` response.

## Expected deliverables

Produce concrete files whenever implementation is requested. A complete model integration usually includes:

- `loongforge/embodied/eval/servers/<model>_policy.py` or an extension of an existing policy adapter.
- `loongforge/embodied/eval/servers/<model>_server.py` if the model cannot reuse an existing server entrypoint.
- `loongforge/embodied/eval/orchestrator/server_manager.py` routing so `model.backend: <model>` starts the right server.
- Demo YAML configs for every already-supported benchmark under `examples/embodied/<model>/eval/configs/<benchmark>/`, unless the user explicitly narrows scope.
- Optional run scripts under `examples/embodied/<model>/eval/` if this project already has scripts for that model/benchmark pattern.
- A smoke-test matrix covering every generated benchmark demo, with one command and expected artifact path per benchmark.
- Executed smoke tests for every generated benchmark demo that can run in the current environment; do not stop at generating YAML/matrix files.
- Smoke outputs under `loongforge/embodied/eval/reports/<model>/<benchmark>/...` for every benchmark that can be executed in the current environment.
- Documentation updates in `loongforge/embodied/eval/benchmark_envs.md`, `user_guide.md`, or `README.md`.

If the user asks for a dry-run or generation test, write generated artifacts to a temp directory and do not overwrite repo files.

## Workflow

1. Inspect the target model and benchmark context.
   - Find existing model configs, server entries, and policy adapters.
   - Identify whether this is a new backend or a variant of an existing backend.
   - Confirm target benchmarks are already wired in the orchestrator.

2. Define the model policy adapter boundary.
   - Accept the unified RPC fields used by runners: `images`, `instruction`, `episode_id`, `episode_step`, `disable_action_cache`, `return_action_chunk`, `cfg_scale`, and optional state fields.
   - Convert image/state/instruction inputs to the model's processor/tokenizer/inference format.
   - Return a dict containing at least `actions`; include latency/metadata if local patterns do.
   - Keep benchmark-specific obs/action conversion out of the model policy.

3. Handle action shape and normalization explicitly.
   - Record `action_dim`, `state_dim`, `action_horizon`, `max_action_dim`, and `max_state_dim` in YAML.
   - Check whether the model emits raw actions or normalized actions.
   - If normalized, require a matching `dataset_statistics_path` and implement the correct inverse transform.
   - Do not reuse pi05 q01/q99 unnormalization for another model unless the model uses that exact convention.
   - Validate action chunks can reshape to the benchmark action dimension, such as 7D single-arm or 14D bimanual.

4. Write YAML configs for all supported benchmarks.
   - Discover the already-supported benchmark set from existing runner/config directories before generating examples.
   - By default, generate one short demo/smoke YAML for each supported benchmark, such as LIBERO, CALVIN, SimplerEnv, RoboTwin, and ManiSkill. Do not use one benchmark as a substitute for all-benchmark coverage unless the user explicitly narrows scope.
   - Keep demo YAMLs bounded for smoke validation: prefer one task, one episode, and low max steps where the runner supports those knobs. Do not copy a full multi-task or long-horizon evaluation template and call it a smoke test.
   - When deriving a smoke YAML from an existing eval YAML, actively reduce it to smoke size instead of copying the source unchanged. Use local runner knobs where available: LIBERO `max_tasks: 1` and `episodes_per_task: 1`; CALVIN one sequence and a low `max_steps` such as 30; SimplerEnv one task/episode with bounded max steps; RoboTwin one generated episode or one task with bounded max steps; ManiSkill one episode with a low `max_steps` such as 5.
   - Include `model.backend`, `model.model_type`, `model.name`, `model.ckpt_path`, `model.dataset_statistics_path`, action/state dimensions, and runtime knobs.
   - Include `server.python`, `server.host`, `server.port`, `server.health_port`, `server.start_timeout_sec`, and `server.log`.
   - Use unique ports per smoke config to avoid health-port collisions during repeated runs.
   - Include `run.output_dir`, `run.seed`, `run.episode_idx`, `run.save_trace`, and replay flags when supported.
   - For internal configs, use `_internal.yaml` and keep local absolute paths there rather than in public templates.

5. Wire server startup.
   - Add server-manager routing for the new `model.backend`.
   - Ensure health readiness means the policy object has finished construction and action RPC can start.
   - Use reusable health server binding for short repeated smoke runs when local patterns support it.
   - Before running the smoke matrix, check for leftover orchestrator/policy-server processes and occupied server ports. After any timeout or failed run, verify no policy server remains before continuing to the next benchmark.
   - Keep benchmark client Python env and model server Python env explicit. Run `python -m loongforge.embodied.eval.orchestrator.run` with the benchmark/simulator conda environment, while `server.python` in YAML starts the model server environment. Do not use the model server env as the benchmark runner env unless that benchmark actually runs there.

6. Check runtime-specific traps.
   - For SAPIEN-based benchmarks such as SimplerEnv, RoboTwin, and ManiSkill, verify NVIDIA Vulkan with `vulkaninfo`, not just `nvidia-smi`.
   - Expected Vulkan signal is `deviceName = NVIDIA ...` and `driverName = NVIDIA`; `llvmpipe`/`lavapipe` means visual rollout is not trustworthy.
   - SAPIEN runners must set `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR` before importing SAPIEN/svulkan2/ManiSkill; re-exec if needed.
   - For MuJoCo/LIBERO/CALVIN, preserve existing `MUJOCO_GL`, `PYOPENGL_PLATFORM`, and benchmark config-path patterns.

7. Validate in layers by running the generated smoke matrix.
   - Execute every generated benchmark demo that can run in the current environment. The integration is not verified by YAML generation alone.
   - Use the benchmark client's conda environment for the top-level orchestrator command: LIBERO with the LIBERO env, CALVIN with the CALVIN env, SimplerEnv with the SimplerEnv env, RoboTwin with the RoboTwin env, ManiSkill with the ManiSkill env. Let YAML `server.python` start the model server env separately.
   - Mark a benchmark `passed` only when the command exits successfully and the expected artifacts are present. At minimum, check that `policy_server.log` exists, `results.jsonl` exists and contains at least one record, `summary.csv` exists when the runner supports it, and trace/replay artifacts exist when the YAML requested `save_trace` or `save_replay`.
   - Mark a benchmark `blocked` when required runtime, simulator assets, checkpoint, stats, or environment support is missing; include the concrete error or missing path.
   - Mark a benchmark `skipped` only when the user explicitly narrows scope or asks not to run it.
   - Protocol or mock smoke proves runner/server/RPC/action shape.
   - Random-init smoke proves the real model class can initialize and answer RPC, but is not a benchmark score.
   - Real-checkpoint smoke proves the real checkpoint can run one short episode.
   - Credible score requires matching checkpoint, matching `dataset_statistics.json`, correct action semantics, and benchmark-scale episodes.

8. Update docs with precise status.
   - Separate `mock`, `random-init`, `real checkpoint`, and `credible score` statuses.
   - Mention missing assets directly, especially checkpoint and `dataset_statistics.json`.
   - Record any runtime requirements that future users must set before import.

## Pi05 reference mapping

Use pi05 as the canonical example of this structure, while remembering it predates the cleaner naming:

- Model policy adapter: `loongforge/embodied/eval/servers/loongforge_policy.py` with `LoongForgePI05Policy`.
- Server entrypoint: `loongforge/embodied/eval/servers/loongforge_server.py`.
- Routing: `loongforge/embodied/eval/orchestrator/server_manager.py` maps `loongforge`, `pi05`, and `loongforge_pi05` to the LoongForge server.
- YAML configs: `examples/embodied/pi05/eval/configs/<benchmark>/*.yaml`.
- Reports: `loongforge/embodied/eval/reports/pi05/<benchmark>/...`.

For new models, prefer explicit names such as `<model>_policy.py`, `<model>_server.py`, `model.backend: <model>`, and `examples/embodied/<model>/...` so backend, architecture, and product names do not collapse into one concept.

## Required final response

When finished, report:

- Files created or modified.
- The discovered supported benchmark set and whether each benchmark received a demo YAML.
- A per-benchmark smoke matrix with status: passed, skipped, or blocked.
- Which smoke layer passed for each benchmark: mock, random-init, real checkpoint, or credible score.
- Exact command used for each validation that ran.
- Output artifact path such as `results.jsonl` or `policy_server.log` for each completed smoke.
- Any remaining blocker, especially missing checkpoint, missing `dataset_statistics.json`, action mismatch, runtime driver issue, missing simulator env, or user-narrowed scope.
