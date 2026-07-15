---
name: vla-model-eval-adapter
description: Use this skill when adapting a VLA/model backend into the LoongForge embodied eval system after benchmark runners already exist. Trigger on requests to add or refactor model eval integration, implement or validate a model predict_action interface, create model factory/loader code, wire model.backend/server routing, write eval YAML, handle action normalization or dataset_statistics, or reproduce pi05-style integration. Prefer the shared predict_action contract plus GenericPredictActionPolicy before creating a bespoke policy adapter. By default, generate demo YAMLs and smoke-test plans for every already-supported benchmark, such as LIBERO, CALVIN, SimplerEnv, RoboTwin, and ManiSkill, unless the user narrows scope.
---

# VLA Model Eval Adapter

Use this skill to connect a model backend to the existing LoongForge embodied evaluation stack. The benchmark runners and benchmark adapters are assumed to already exist. The preferred model-side architecture is now:

```text
model factory/loader
  -> model instance exposing predict_action(images, instructions, state=None, dataset_stats=None)
  -> GenericPredictActionPolicy
  -> PolicyServer RPC
```

Create a bespoke policy adapter only when the model cannot reasonably expose the shared `predict_action` interface or needs custom RPC behavior that `GenericPredictActionPolicy` cannot cover.

## Mental model

Keep three boundaries separate:

```text
benchmark adapter:
  benchmark obs/action <-> canonical eval schema
  owns benchmark-native state, action conversion, and debug/trace metadata
  examples: adapters/libero.py, adapters/maniskill.py, adapters/simplerenv.py

model factory/loader:
  model config/import/checkpoint/tokenizer/device/dtype/random-init/metadata
  returns a model object implementing predict_action(...)
  example: PI05ModelFactory in servers/loongforge_policy.py

generic eval policy:
  RPC payload handling, image view selection (primary/wrist up to 2 views),
  predict_action invocation, action chunk cache, action shape validation,
  action dim truncation, dataset statistics loading, latency, metadata.
  Action unnormalization is NOT done here; it is the model's responsibility
  inside predict_action().
  example: GenericPredictActionPolicy in servers/loongforge_policy.py
```

The adapter state boundary matters. Benchmark-native structured state stays in `canonical_obs["state"]` for eval/debug/trace. Only model-ready state goes into `canonical_obs["model_state"]`, which runners forward as RPC payload `state`, and which eventually reaches `predict_action(state=...)`.

```text
adapter.model_state -> RPC payload.state -> predict_action(state=...)
```

Do not clean, drop, or reinterpret benchmark-native dict state inside a model factory. That belongs in the benchmark adapter or payload boundary.

## Expected deliverables

Produce concrete files whenever implementation is requested. A complete model integration usually includes:

- A model factory/loader that returns a model instance and metadata, preferably via `PredictActionModelSpec` or an equivalent local pattern.
- A model object exposing `predict_action(images, instructions, state=None, dataset_stats=None)`.
- Interface validation using `validate_predict_action_model()` and `call_predict_action()` from `loongforge/embodied/eval/servers/predict_action_interface.py`.
- Reuse of `GenericPredictActionPolicy` when the shared interface is sufficient.
- A bespoke `servers/<model>_policy.py` only if shared `predict_action` is not a good fit.
- `loongforge/embodied/eval/servers/<model>_server.py` or existing server entrypoint reuse if applicable.
- `loongforge/embodied/eval/orchestrator/server_manager.py` routing only when adding a new `model.backend` that cannot reuse existing LoongForge routing.
- Demo YAML configs for every already-supported benchmark under `examples/embodied/<model>/eval/configs/<benchmark>/`, unless the user narrows scope.
- Optional run scripts under `examples/embodied/<model>/eval/` if this project already has scripts for that model/benchmark pattern.
- A smoke-test matrix covering every generated benchmark demo, with one command and expected artifact path per benchmark.
- Executed smoke tests for every generated benchmark demo that can run in the current environment; do not stop at generating YAML/matrix files.
- Documentation updates in the relevant eval docs, especially `README.md`, `user_guide.md`, `benchmark_envs.md`, `loongforge_eval_summary.md`, or `predict_action_interface.md`.

If the user asks for a dry-run, generation test, or re-application test, write generated artifacts to a temp directory such as `/tmp/<model>_eval_adapter_*` and do not overwrite repo files except the skill itself when explicitly requested.

## Workflow

1. Inspect the target model and benchmark context.
   - Find existing model configs, server entries, policy adapters, and model inference APIs.
   - Identify whether this is a new backend, a variant of an existing backend, or a refactor of an existing integration.
   - Confirm target benchmarks are already wired in the orchestrator.
   - Discover the already-supported benchmark set from runner/config directories rather than assuming only one benchmark.

2. Decide whether the shared `predict_action` path applies.
   - Prefer adding or reusing a model method with this shape:

     ```python
     def predict_action(images, instructions, state=None, dataset_stats=None):
         ...
     ```

   - Use `PredictActionModel`, `validate_predict_action_model()`, and `call_predict_action()` to define and test the contract.
   - Accept these output shapes from the model and normalize to `[H, action_dim]`: `[D]`, `[H, D]`, or `[B, H, D]`.
   - Make the model factory handle model-private setup: imports, config registration, checkpoint loading, tokenizer paths, device/dtype, compile flags, random-init, and metadata.
   - Let `GenericPredictActionPolicy` handle eval-private behavior: RPC payloads, image view selection (extracts `primary`/`head` plus one of `wrist`/`right`/`left`, maximum 2 views), chunk caching, action shape validation, latency, request IDs, and response format. Action unnormalization is the model's responsibility and must happen inside `predict_action()`.

3. Define state and action semantics explicitly.
   - Keep benchmark-native `canonical_obs["state"]` out of the model server unless it is already model-ready.
   - Add or preserve `canonical_obs["model_state"]`; runners should forward `canonical_obs.get("model_state")` as RPC payload `state`.
   - If a model consumes state, verify that `model_state` ordering, units, frame, shape, and `dataset_stats["observation.state"]` match training.
   - Record `action_dim`, `state_dim`, `action_horizon`, `max_action_dim`, and `max_state_dim` in YAML.
   - Check whether the model emits raw actions or normalized actions.
   - If normalized actions require inverse transform (e.g. pi05 q01/q99, or LeRobot mean/std), implement that inside the model's `predict_action()`, not in the generic eval policy.
   - Do not reuse pi05 q01/q99 unnormalization for another model unless the model uses that exact convention.

4. Write YAML configs for all supported benchmarks.
   - By default, generate one short demo/smoke YAML for each supported benchmark, such as LIBERO, CALVIN, SimplerEnv, RoboTwin, and ManiSkill.
   - Keep demo YAMLs bounded: prefer one task, one episode, and low max steps where runner knobs allow it.
   - Use local runner knobs where available: LIBERO `max_tasks: 1` and `episodes_per_task: 1`; CALVIN one sequence and low max steps; SimplerEnv one task/episode with bounded max steps; RoboTwin one generated episode or one task with bounded max steps; ManiSkill one episode with low `max_steps`.
   - Include `model.backend`, `model.model_type`, and model-structure fields (`action_dim`, `action_horizon`, etc.) in the `model:` section.
   - Include infrastructure fields (`ckpt_path`, `tokenizer_path`, `dataset_statistics_path`, `use_bf16`, `loongforge_root`) in the `server:` section, not `model:`.
   - Include `server.python`, `server.host`, `server.port`, `server.health_port`, `server.start_timeout_sec`, and `server.log`.
   - Use unique ports per smoke config to avoid health-port collisions during repeated runs.
   - Include `run.output_dir`, `run.seed`, `run.save_trace`, and replay flags when supported.
   - For internal configs, use `_internal.yaml` and keep local absolute paths there rather than in public templates.

5. Wire server startup only as needed.
   - Reuse existing LoongForge server routing for pi05-style integrations when possible.
   - Add server-manager routing for a new `model.backend` only when an existing server entrypoint cannot serve it.
   - Ensure health readiness means the model factory has completed, the warmup `predict_action()` call has run (see below), and action RPC can start.
   - Use reusable health server binding for short repeated smoke runs when local patterns support it.
   - After model factory build and before health server startup, run a warmup `predict_action()` with a zero-filled dummy image and an empty instruction. This forces all lazy imports (including potential circular-import paths) to complete before the first real episode arrives. The call is wrapped in a try/except so a warmup exception does not abort startup, but the model must not enter a corrupted state from a warmup call.
   - Before running the smoke matrix, check for leftover orchestrator/policy-server processes and occupied server ports.
   - Keep benchmark client Python env and model server Python env explicit. Run the top-level orchestrator with the benchmark/simulator conda environment, while YAML `server.python` starts the model server environment.

6. Check runtime-specific traps.
   - For SAPIEN-based benchmarks such as SimplerEnv, RoboTwin, and ManiSkill, verify NVIDIA Vulkan with `vulkaninfo`, not just `nvidia-smi`, when visual rollout correctness matters.
   - Expected Vulkan signal is `deviceName = NVIDIA ...` and `driverName = NVIDIA`; `llvmpipe`/`lavapipe` means visual rollout is not trustworthy.
   - SAPIEN runners may need `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR` set before importing SAPIEN/svulkan2/ManiSkill.
   - For MuJoCo/LIBERO/CALVIN, preserve existing `MUJOCO_GL`, `PYOPENGL_PLATFORM`, and benchmark config-path patterns.

7. Validate in layers.
   - First run local interface validation without a benchmark when possible:

     ```bash
     PYTHONPATH=/workspace/LoongForge-VLA python - <<'PY'
     import numpy as np
     from loongforge.embodied.eval.servers.predict_action_interface import call_predict_action, validate_predict_action_model

     class MyModel:
         def predict_action(self, images, instructions, state=None, dataset_stats=None):
             return np.zeros((len(instructions), 4, 7), dtype=np.float32)

     model = MyModel()
     validate_predict_action_model(model)
     print(call_predict_action(model, images=[[]], instructions=["task"], state=None, dataset_stats=None, action_dim=7).shape)
     PY
     ```

   - Then execute every generated benchmark demo that can run in the current environment.
   - Use the benchmark client's conda environment for the top-level orchestrator command: LIBERO with the LIBERO env, CALVIN with the CALVIN env, SimplerEnv with the SimplerEnv env, RoboTwin with the RoboTwin env, ManiSkill with the ManiSkill env.
   - Mark a benchmark `passed` only when the command exits successfully and the expected outputs prove at least one policy call or official runner completion.
   - Mark a benchmark `blocked` when required runtime, simulator assets, checkpoint, stats, or environment support is missing; include the concrete error or missing path.
   - Mark a benchmark `skipped` only when the user explicitly narrows scope or asks not to run it.
   - Protocol or mock smoke proves runner/server/RPC/action shape.
   - Random-init smoke proves the real model class can initialize and answer RPC, but is not a benchmark score.
   - Real-checkpoint smoke proves the real checkpoint can run one short episode.
   - Credible score requires matching checkpoint, matching `dataset_statistics.json`, correct action semantics, and benchmark-scale episodes.

8. Update docs with precise status.
   - Separate `mock`, `random-init`, `real checkpoint`, and `credible score` statuses.
   - Put user-facing usage in README/user guide; avoid filling README with internal validation logs.
   - Put detailed interface contract and local interface-validation examples in `predict_action_interface.md`.
   - Mention missing assets directly, especially checkpoint and `dataset_statistics.json`.
   - Record runtime requirements that future users must set before import.

## Pi05 reference mapping

Use pi05 as the canonical example of the shared `predict_action` architecture:

- Model interface: `PI05Policy.predict_action(images, instructions, state=None, dataset_stats=None)` in the model package.
- Interface helpers: `loongforge/embodied/eval/servers/predict_action_interface.py`.
- Generic eval policy: `GenericPredictActionPolicy` in `loongforge/embodied/eval/servers/loongforge_policy.py`.
- Model factory: `PI05ModelFactory` in `loongforge/embodied/eval/factories/pi05_factory.py`.
- Factory registry: `register_factory`, `build_model_spec` in `loongforge/embodied/eval/factories/registry.py`. New models register with `@register_factory("<model_type>")` and declare `model_config_cls = <ModelConfig>`.
- Server config: `EvalServerArgs` dataclass and `parse_eval_server_config` in `loongforge/embodied/eval/servers/eval_server_config.py`. The YAML `server:` section is merged directly into `EvalServerArgs` via OmegaConf; the YAML `model:` section is merged into the registered `ModelConfig` (e.g. `Pi05ModelConfig`) via OmegaConf.
- Backward-compatible wrapper: `LoongForgePI05Policy` in `loongforge/embodied/eval/factories/pi05_factory.py`, which should not be the preferred pattern for new integrations.
- Server entrypoint: `loongforge/embodied/eval/servers/loongforge_server.py` calls `parse_eval_server_config` to get `EvalServerArgs` + `raw_model_dict`, then `build_model_spec` to load the model, then `_warmup_model()` to resolve lazy imports, then wraps in `GenericPredictActionPolicy`.
- Routing: `loongforge/embodied/eval/orchestrator/server_manager.py` maps `loongforge`, `pi05`, and `loongforge_pi05` to the LoongForge server.
- Adapter state boundary: benchmark adapters provide `canonical_obs["state"]` for native structured state and `canonical_obs["model_state"]` for model-ready state; runners forward only `model_state` as RPC payload `state`.
- YAML configs: `examples/embodied/pi05/eval/configs/<benchmark>/*.yaml`. Infrastructure fields (`ckpt_path`, `tokenizer_path`, `dataset_statistics_path`, `use_bf16`, `loongforge_root`) live in `server:` section; model-structure fields (`action_dim`, `action_horizon`, `compile_model`, etc.) live in `model:` section.

For new models, create `loongforge/embodied/eval/factories/<model>_factory.py` with a `@register_factory("<model_type>")` class that declares `model_config_cls` and implements `build(model_cfg, server_args) -> PredictActionModelSpec`. Add the module path to `_FACTORY_MODULES` in `factories/registry.py`. No changes to `loongforge_server.py` are needed.

## Required final response

When finished, report:

- Files created or modified.
- Whether the integration used the shared `predict_action` path or a bespoke policy adapter, and why.
- The discovered supported benchmark set and whether each benchmark received a demo YAML, unless the user narrowed scope.
- A per-benchmark smoke matrix with status: passed, skipped, or blocked.
- Which smoke layer passed for each benchmark: local interface validation, mock, random-init, real checkpoint, or credible score.
- Exact command used for each validation that ran.
- Output artifact path such as `results.jsonl` or `policy_server.log` for each completed smoke, or note if temp artifacts were deleted after validation.
- Any remaining blocker, especially missing checkpoint, missing `dataset_statistics.json`, action mismatch, runtime driver issue, missing simulator env, or user-narrowed scope.
