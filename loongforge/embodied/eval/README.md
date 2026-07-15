# LoongForge-VLA Offline Eval

This directory contains the LoongForge-VLA offline evaluation module. It runs benchmark clients and model policy servers as separate processes connected by a WebSocket/msgpack-numpy RPC protocol.

## Scope

- `loongforge.embodied.eval.protocol`: canonical observation/action/result schema.
- `loongforge.embodied.eval.transport`: WebSocket RPC client/server utilities.
- `loongforge.embodied.eval.adapters`: benchmark-side adapters.
- `loongforge.embodied.eval.servers.loongforge_server`: LoongForge policy server entrypoint.
- `loongforge.embodied.eval.servers.loongforge_policy`: generic `predict_action` eval policy (`GenericPredictActionPolicy`) and shared data types.
- `loongforge.embodied.eval.servers.eval_server_config`: `EvalServerArgs` dataclass and `parse_eval_server_config` YAML parser.
- `loongforge.embodied.eval.servers.predict_action_interface`: shared model interface checks and action shape normalization.
- `loongforge.embodied.eval.servers.mock_policy`: lightweight protocol mock for tests.
- `loongforge.embodied.eval.factories.registry`: model factory registry (`MODEL_FACTORY_REGISTRY`, `register_factory`, `build_model_spec`).
- `loongforge.embodied.eval.factories.pi05_factory`: PI05 model factory (`PI05ModelFactory`).

LoongForge source code under the repo root `loongforge/` is not patched by this eval module. Model-specific compatibility lives in `eval/factories`.

## Quick Start: pi05 on LIBERO / CALVIN / SimplerEnv / RoboTwin / ManiSkill

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
- `examples/embodied/pi05/eval/configs/maniskill/pick_cube_20step.yaml`: ManiSkill 7D single-arm PickCube smoke example.

The pi05 configs use:

- `model.backend: loongforge`
- `model.model_type: pi05`
- `model.action_dim`, `model.action_horizon`, etc.: Pi05ModelConfig structure fields
- `server.ckpt_path`: checkpoint directory or `model.safetensors`, unless `server.random_init: true` is set for a smoke run
- `server.dataset_statistics_path`: dataset stats passed through to model `predict_action()` for model-owned normalization or unnormalization
- `server.python`: LoongForge Python environment
- `run.output_dir`: runtime output directory under `eval/reports/<model>/<benchmark>/<run_name>/`; `reports/` is generated locally and is not committed

For protocol/debug smoke testing, copy an existing YAML and set `model.backend: mock`; mock backend configs are not shipped as pi05 examples because they are not model-specific.

## Model Interface

LoongForge model servers now prefer a shared `predict_action` interface instead of a separate full policy adapter for every model. A reusable `GenericPredictActionPolicy` handles eval RPC behavior: canonical image view selection, `predict_action` invocation, action shape validation, action-dim truncation, chunk caching, latency reporting, metadata, and dataset statistics loading. Model-specific normalization and unnormalization should happen inside the model's `predict_action()` implementation.

Model-specific logic should be kept in a thin factory under `eval/factories/`. The current pi05 path is:

```text
loongforge_server.py
  -> parse_eval_server_config(yaml)      # EvalServerArgs + raw_model_dict
  -> build_model_spec(server_args, raw_model_dict)
       -> build_model_config("pi05", raw_model_dict)  # OmegaConf → Pi05ModelConfig
       -> PI05ModelFactory.build(model_cfg, server_args)
            -> build_model(model_cfg)                 # training-side registry
            -> model.model.load_pretrained(ckpt_path) # training-side weight loader
            -> model.to(device).eval()
  -> _warmup_model(model_spec)           # dummy predict_action() to resolve lazy imports
  -> GenericPredictActionPolicy(...)
  -> PI05Policy.predict_action(images, instructions, state=None, dataset_stats=None)
```

To add a new model, create `eval/factories/<model>_factory.py`, implement a factory class with `model_config_cls` and a `build(model_cfg, server_args)` classmethod, decorate it with `@register_factory("<model_type>")`, and add its module path to `_FACTORY_MODULES` in `eval/factories/registry.py`. No changes needed in `loongforge_server.py`.

## Outputs

A benchmark run writes:

- `results.jsonl`: one JSON record per episode
- `summary.csv`: task-level summary
- `suite_summary.csv`: suite-level summary
- `artifacts/.../replay_*.gif`: optional replay for direct rollout runners such as LIBERO and SimplerEnv
- `artifacts/.../trace_*.json` or `artifacts/.../trace.json`: optional per-step action trace
- `artifacts/.../videos/*.mp4`: optional RoboTwin official evaluator videos when video logging is enabled
- `policy_server.log`: model server stdout/stderr

## Development Notes

- Keep model framework logic in `eval/factories/<model>_factory.py`. Do not add model-specific code to `loongforge_server.py`.
- Do not modify `../loongforge` source code for eval-specific compatibility.
- Prefer adding YAML configs over command-line parameter sprawl.
- Use `model.backend` in YAML for backend selection and `benchmark.name` for benchmark selection.
- New model factories should ensure their `predict_action()` tolerates a warmup call with a zero-filled dummy image and an empty instruction string (`_warmup_model` in `loongforge_server.py` runs this before serving).
