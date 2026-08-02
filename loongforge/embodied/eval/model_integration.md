# Guide to Integrating a New Model into the Eval System (including the `predict_action` contract, using pi05 and xvla as examples)

This document summarizes the complete workflow, key configuration items, and the `predict_action` interface contract that the model side must implement in order to integrate a new VLA model into the `loongforge/embodied/eval` eval system.
Beyond implementing a Factory + PayloadBuilder (+ an optional ActionDecoder) and writing the YAML config, integration also involves a series of model-semantics-level configuration points that must be confirmed one by one.
pi05 and xvla differ almost entirely on these configuration points, so this document uses the two as examples to walk through each item and provides a configuration comparison table.

**Scope note:** The semantic checklist below covers all benchmarks currently integrated (LIBERO / RoboTwin / SimplerEnv / CALVIN / ManiSkill).
Early integration used LIBERO as the primary acceptance path, so some items are illustrated with LIBERO; cross-benchmark protocol differences (such as RoboTwin `action_bridge`) are called out separately.
For the config layout conventions see §1; for task success status see `README.md` / `user_guide_en.md`.

Overall architecture (a four-stage chain that decouples the model from the benchmark):

```text
Adapter (obs -> canonical, exposes state_raw)
  -> PayloadBuilder (canonical -> predict_action kwargs, per model, capabilities declared via class attributes)
  -> model.predict_action (policy server, server side)
  -> ActionDecoder (raw chunk -> env action; key auto-assembled as {action_encoding}_to_{action_space}, identity -> IdentityDecoder)
```

---

## 1. Overview (three-part integration)

Integrating a new model = **Factory + PayloadBuilder + optional ActionDecoder**, with the three separated by responsibility.
Add new files, and **do not modify existing files**:

1. **ModelFactory (server side, requires torch)** — add `<model>_factory.py` under the `factories/` directory,
   register it with `@register_factory("<model_type>")`, and have the `build()` method return
   `PredictActionModelSpec(model, metadata)`. It is responsible for import, config, weight loading / `random_init`,
   device/dtype, and wrapping the model into an object that implements the unified `predict_action(images, instructions, state=None,
   dataset_stats=None, **kwargs)` interface (interface contract see §2).
2. **PayloadBuilder (client side)** — add `<model>.py` under the `payload_builders/` directory,
   register it with `@register_payload_builder("<model_type>")`, and inherit from the `PayloadBuilder` base class.
   Use **type-annotated class attributes** to declare capabilities (`state_encoding` / `action_encoding` / `action_dim` /
   `action_horizon` / `domain_id` / `unnorm_key`, etc.; YAML can override fields of the same name);
   `build(canonical, ctx)` converts the canonical dict into the kwargs for `predict_action`.
   Optionally implement `reset` / `update_from_response` / `note_env_action` to handle closed-loop feedback.
3. **ActionDecoder (optional, eval side)** — only when the model's `action_encoding` does not yet have a corresponding decoder,
   add / reuse a decoder under the `action_decoders/` directory. The decoder key is
   **automatically assembled** by the orchestrator as `{action_encoding}_to_{action_space}` (see §4): when the source encoding == target space,
   an empty key is returned → `IdentityDecoder` passthrough; when it hits an already-registered key (such as some `ee6d_*` decoder), it is selected automatically,
   with no new code needed.
4. Write the eval config and scripts (`examples/embodied/<model>/eval/`); the model-specific YAML fields
   (`state_encoding` / `action_encoding` / `domain_id`, etc.) go under the `model:` section,
   and verify each configuration point listed in sections 4 and 5 of this document one by one.
   `model.model_type` is **required** (no default): `EvalServerArgs.model_type` carries no fallback and
   `parse_eval_server_config` raises when the `model:` section omits it.
5. First run a smoke (1 task × 1 episode, or a chain smoke) to validate the RPC / action semantics,
   then, when domain weights are available, do task-success and full-scale eval.
6. After modifying **shared** code such as the runner / adapter / decoder / bridge / generic policy, run a regression on
   already-succeeding combinations (at least pi05×LIBERO) to avoid the new protocol breaking existing results.

> **Startup consistency check:** the orchestrator asserts
> `set(MODEL_FACTORY_REGISTRY) == set(PAYLOAD_BUILDER_REGISTRY)`.
> The Factory and PayloadBuilder model_type must exist in pairs; writing only half causes a fail-fast.

### 1.1 Config and script layout (pi05 / xvla convention)

- Keep **only one pair** of YAML files per benchmark: a public template `*.yaml` (with `/path/to/...` placeholders) +
  an internal one-click `*_internal.yaml` (with local absolute paths). Optional knobs go in the header comments of the file.
  **Do not** add extra full-suite / backup smoke files (unless the user explicitly requests them).
- Scripts come in pairs: `run_<benchmark>_eval.sh` ↔ public YAML;
  `run_<benchmark>_eval_internal.sh` ↔ `_internal` YAML.
- **task-success vs chain smoke:** only write task-success when matching domain weights exist and have been validated;
  when there are no domain weights, use `server.random_init: true`, an empty `ckpt_path`, and a short `max_steps`,
  and mark **not task-success** in the comments (see the CALVIN and ManiSkill of pi05/xvla;
  and the SimplerEnv of pi05).

Reference directories:

- `examples/embodied/pi05/eval/configs/<benchmark>/`
- `examples/embodied/xvla/eval/configs/<benchmark>/`

---

## 2. The `predict_action` contract (reference)

This section is the interface contract between the **model author** and the eval stack, aimed at the model owner who implements `predict_action()`.
The **single source of truth** for the helper functions is `loongforge/embodied/eval/servers/predict_action_interface.py`:
after refactoring, this file **only retains the model-author contract** — the `PredictActionModel` protocol, `validate_predict_action_model`,
`_filter_supported_kwargs`, and `call_predict_action`; it **does not contain** any action-space decoding logic,
which has been moved to `loongforge/embodied/eval/action_decoders/` (see §4).

### 2.1 Responsibility boundaries of each layer

| Layer | Owner | Responsibility |
|---|---|---|
| `predict_action(...)` | **Model** | Inference; optional **state normalization** / **action denormalization** (using `dataset_stats`); returns a numeric action chunk |
| Model factory | eval `factories/` | import, config, ckpt / `random_init`, device/dtype; wraps into a callable `predict_action` object |
| `PayloadBuilder` | eval `payload_builders/` | Client side: canonical dict → `predict_action` kwargs (image packing, state encoding, model-specific fields such as `domain_id` / `unnorm_key`) |
| `GenericPredictActionPolicy` | eval server | RPC, action chunk caching, latency, shape checking, **dim truncation** to `action_dim`; wraps the PayloadBuilder's view list into a batch-of-1 `[[...]]` |
| `ActionDecoder` (`action_decoders/`) | eval runner / bridge | Model action encoding → **env** action space (e.g. 20D ee6d → 7D LIBERO); does **not** perform dataset q99 denormalization |

**Do not** write LIBERO / RoboTwin / SimplerEnv environment special-casing into the training tree's `predict_action`.
When the model action space is inconsistent with the environment, prefer a named eval `ActionDecoder` / bridge.

### 2.2 Recommended integration path

Once a benchmark runner already exists, **do not** fork another full `LoongForgeXXXPolicy`.
(The backward-compatibility wrappers `LoongForgePI05Policy` / `LoongForgeXVLAPolicy` still exist, but are **not** the preferred path.)
Model integration should follow the three-part workflow in §1. The server side is uniformly handled by `GenericPredictActionPolicy` for RPC / caching /
statistics path / shape checking; at startup the orchestrator asserts
`set(MODEL_FACTORY_REGISTRY) == set(PAYLOAD_BUILDER_REGISTRY)`, and a Factory missing its paired PayloadBuilder causes a fail-fast.

Reference paths already in the repo:

```text
loongforge_server.py
  -> PI05ModelFactory.build(...)              # factories/pi05_factory.py
  -> GenericPredictActionPolicy(...)
  -> model.predict_action(images, instructions, state=None, dataset_stats=None, **kwargs)

loongforge_server.py
  -> XVLAModelFactory.build(...)              # factories/xvla_factory.py
  -> GenericPredictActionPolicy(...)
  -> wrapper.predict_action(images, instructions, state=None, dataset_stats=None, domain_id=..., **kwargs)
```

The standard signature is `predict_action(images, instructions, state=None, dataset_stats=None, **kwargs)`.
Extra kwargs emitted by the PayloadBuilder (such as `domain_id`, `unnorm_key`) are passed through when the signature accepts them;
unknown kwargs the signature does not accept are dropped by `_filter_supported_kwargs` with a WARNING (see §2.4).

### 2.3 Required signature

```python
def predict_action(images, instructions, state=None, dataset_stats=None, **kwargs):
    """Return decoder-ready actions as a float array."""
    ...
```

Parameter meanings:

| Parameter | Typical type | Meaning |
|---|---|---|
| `images` | a batch-of-1 list-of-list (each element is a view array) | e.g. `[[view_primary, view_wrist]]`. The PayloadBuilder packs the per-camera dict into a view list (prefer `primary`/`head`; if `left` and `right` both exist → `[primary, left, right]`; otherwise append at most one more `wrist` → `[primary, wrist]`; single view → `[primary]`); the server then wraps it into batch-of-1. |
| `instructions` | `list[str]` | Batched language instructions (batch=1). |
| `state` | `None` or a numeric vector / array | The **model-usable** proprio the PayloadBuilder produces from `canonical["state_raw"]` according to `state_encoding` (may be `None`). |
| `dataset_stats` | `dict` or `None` | Loaded by the eval side from `server.dataset_statistics_path` and **passed through**; used internally by the model as needed for state normalization / action denormalization. |
| Extra kwargs | e.g. `domain_id`, `unnorm_key` | Emitted by the PayloadBuilder. Passed through only when the signature accepts them (explicitly named or `**kwargs`); unknown ones are dropped with a WARNING. |

Images are a batch-of-1 list-of-list (each inner element is a view array), and instructions are `list[str]`.
They are delivered to the model via the RPC v2 payload.

### 2.4 kwarg filtering (`_filter_supported_kwargs`)

`call_predict_action` inspects the signature before calling:

- Signature contains `**kwargs` → all extra keywords are passed through.
- Fixed signature → only the declared parameters are kept. Any field the PayloadBuilder emits but the model does not declare (such as a mistakenly emitted
  `domain_id` / `unnorm_key`) is dropped and a **WARNING is logged**, making a misconfigured PayloadBuilder
  visible in the logs rather than producing a "no error but wrong result" run.

If a model needs to consume a certain field, it must declare that parameter in the signature (or use `**kwargs`).

### 2.5 Where `state` comes from

The adapter **no longer encodes** proprio; it only exposes the raw fields in `canonical["state_raw"]`
(`eef_pos` / `eef_quat` / `ee_ori_mat` / `joint` / `endpose` / `robot_obs`, etc.);
**each model's PayloadBuilder** encodes them into the `state` kwarg according to its `state_encoding`
(`""` → no state; `ee6d` / `aloha_pi` / `passthrough`, etc.).

The PayloadBuilder's `build(canonical, ctx)` returns the model kwargs; the runner adds RPC control fields around them:

```python
model_kwargs = payload_builder.build(canonical_obs, ctx)  # {"images": [...], "instructions": [...], "state": ...}
rpc = {"episode_id": ..., "episode_step": ..., "disable_action_cache": ..., "return_action_chunk": ...}
rpc.update(model_kwargs)
```

```text
canonical.state_raw  ->  PayloadBuilder(state_encoding)  ->  RPC payload.state  ->  predict_action(state=...)
```

### 2.6 Validation helpers (eval code)

```text
loongforge/embodied/eval/servers/predict_action_interface.py
```

| API | Purpose |
|---|---|
| `PredictActionModel` | Protocol |
| `validate_predict_action_model(model)` | Pre-call signature check |
| `_filter_supported_kwargs(func, kwargs)` | Drops kwargs the signature does not accept (one WARNING logged per dropped item) |
| `call_predict_action(model, images, instructions, state, dataset_stats, action_dim, **kwargs)` | validate → filter kwargs → call → reshape / truncate to `[H, action_dim]` |

`GenericPredictActionPolicy` always calls through `call_predict_action`. Action-space decoding is **not here**; it is in `action_decoders/`.

`validate_predict_action_model` checks: `predict_action` exists and is callable; the required parameters `images` / `instructions` are present;
the optional parameters `state` / `dataset_stats` can be accepted (as named parameters or via `**kwargs`).

```python
# invalid: missing instructions, and cannot accept state/dataset_stats
def predict_action(self, images):
    ...

# valid
def predict_action(self, images, instructions, state=None, dataset_stats=None):
    ...

def predict_action(self, images, instructions, **kwargs):
    state = kwargs.get("state")
    dataset_stats = kwargs.get("dataset_stats")
    ...
```

### 2.7 Action output contract

The model may return `[D]` / `[H, D]` / `[B, H, D]`; `call_predict_action` uniformly normalizes to `[H, action_dim]`:

| Input shape | Behavior |
|---|---|
| `[D]` | → `[1, D]` |
| `[H, D]` | Kept as a chunk |
| `[B, H, D]` | → `[-1, D]` (single-request path) |
| Other ndim | `ValueError` |
| Last dim `< action_dim` | `ValueError` |
| Last dim `> action_dim` | **Truncated** to the first `action_dim` columns |

`action_dim` comes from the model / YAML config (e.g. 7 for single-arm, 14 for RoboTwin joints, 20 for xvla ee6d). **Truncation cannot replace correct action semantics.**

**Normalization / denormalization belongs to the model:** the eval does **not** perform q01/q99, mean/std, or min/max outside the model.
If the network outputs normalized actions, they must be denormalized **inside** `predict_action` using `dataset_stats` (and the training normalization mode).
For example, pi05's ACTION quantile normalization uses `dataset_stats["action"].q01` / `.q99`; other LeRobot-family models may use mean/std or min/max.
The return value should be in the **model action space** (i.e. the encoding declared by the PayloadBuilder's `action_encoding`), and either match the environment after truncation
(e.g. pi05 LIBERO 7D `axis_angle`), or be converted by an eval `ActionDecoder` / RoboTwin `action_bridge` (e.g. xvla 20D ee6d).

Action decoding on the eval side does **not** belong to `predict_action`: after the server returns the chunk, the runner applies the decoders in `action_decoders/`:

```python
from loongforge.embodied.eval.action_decoders import build_action_decoder
from loongforge.embodied.eval.orchestrator.config import resolve_action_decoder_key

key = resolve_action_decoder_key(payload_builder, adapter)  # {action_encoding}_to_{action_space}
decoder = build_action_decoder(key)                         # empty key -> IdentityDecoder
env_actions = decoder(raw_chunk, ctx)                       # __call__(actions[H, D], ctx) -> env_actions
```

### 2.8 Local interface validation (no benchmark)

No GPU / weights needed; use this to validate the eval helpers before fully loading the model:

```bash
cd /workspace/LoongForge-VLA
PYTHONPATH=/workspace/LoongForge-VLA python - <<'PY'
import numpy as np
from loongforge.embodied.eval.servers.predict_action_interface import (
    call_predict_action,
    validate_predict_action_model,
)

class FixedSigModel:
    """pi05-style: extra kwargs are filtered out."""

    def predict_action(self, images, instructions, state=None, dataset_stats=None):
        return np.zeros((len(instructions), 4, 7), dtype=np.float32)

class KwargsModel:
    """xvla-style: domain_id is forwarded."""

    def predict_action(self, images, instructions, state=None, dataset_stats=None, **kwargs):
        assert kwargs.get("domain_id") == 6
        return np.zeros((4, 20), dtype=np.float32)

images = [[np.zeros((224, 224, 3), dtype=np.uint8)]]
common = dict(instructions=["pick up the cube"], state=None, dataset_stats=None)

m = FixedSigModel()
validate_predict_action_model(m)
print(call_predict_action(m, images=images, action_dim=7, do_sample=False, cfg_scale=1.5, **common).shape)

m = KwargsModel()
validate_predict_action_model(m)
print(call_predict_action(m, images=images, action_dim=20, domain_id=6, **common).shape)
PY
```

Expected output:

```text
(4, 7)
(4, 20)
```

Comparison of the `predict_action` output of the real pi05 / xvla after loading via factory (`random_init` is enough for a contract check,
but full correctness (unnorm, abs/delta, decode) still requires a YAML smoke / task-success run, see `user_guide_en.md`):

| Model | Raw `predict_action` | After `call_predict_action` |
|---|---|---|
| pi05 | `[B, action_horizon, max_action_dim]`, e.g. `(1, 50, 32)` | truncate last dim → `(50, 7)` |
| xvla | `[B, num_actions, real_action_dim]`, e.g. `(1, 30, 20)` | reshape + keep dim → `(30, 20)` |

- pi05 **requires** `tokenizer_path` (PaliGemma tokenize) even with `random_init`.
- xvla **requires** a valid Florence processor/tokenizer directory (`tokenizer_path`); an empty path causes HF loading to fail.
- The xvla factory converts the `domain_id` int → a `LongTensor` on the device; `call_predict_action(..., domain_id=3)` can just pass a YAML-style int.

### 2.9 Warmup and common errors

Before the health endpoint is ready, the server may make one call first (`images=[[np.zeros((224,224,3), uint8)]]`, `instructions=["warmup"]`, `state=None`, `dataset_stats=None`).
A failure only logs a warning, but this call must not corrupt weights or render the process unusable; prefer a lazy import that can complete safely.

| Error | Cause |
|---|---|
| `TypeError: model must expose a callable predict_action(...)` | Missing method |
| `TypeError: ... missing required parameters: ['instructions']` | Wrong signature |
| `TypeError: ... cannot accept eval keyword parameters: ['state']` | No `state` and no `**kwargs` |
| `ValueError: ... unsupported action shape` | Not `[D]` / `[H,D]` / `[B,H,D]` |
| `ValueError: ... action dim X, expected at least Y` | Output dimension narrower than `action_dim` |
| The env has steps but the success rate is always 0 | Usually the control mode (abs vs delta), a wrong ActionDecoder / bridge, or wrong unnorm — **not** a missing `predict_action` |

Self-check for the model owner before delivery: `predict_action(images, instructions, state=None, dataset_stats=None)` is callable;
`validate_predict_action_model` passes; returns `[D]`/`[H,D]`/`[B,H,D]` with last dim ≥ `action_dim`;
denormalization (if any) happens **inside** `predict_action`, and the eval only passes through `dataset_stats`; warmup is safe;
the factory is responsible for loading weights / tokenizer / processor and does not rewrite the benchmark dict observation.

---

## 3. Integration workflow (detailed walkthrough of the three components Factory / PayloadBuilder / ActionDecoder)

### 3.1 ActionDecoder (eval side, auto-matched)

Components: `action_decoders/` (`base.py` defines the `ActionDecoder` base class + `IdentityDecoder` +
`ACTION_DECODER_REGISTRY`; `ee6d.py` holds the ee6d source-encoding decoders; `joint.py` holds the joint source-encoding decoders;
`rotation.py` stores the rotation math). The orchestrator **automatically assembles** the decoder key from
`{payload_builder.action_encoding}_to_{adapter.action_space}` (`resolve_action_decoder_key`), and when the source encoding == target space it returns an empty key → `IdentityDecoder` passthrough.

Registered keys (`ACTION_DECODER_REGISTRY`) auto-assembled as `{action_encoding}_to_{action_space}`:

| key | Use |
|---|---|
| `ee6d_to_axis_angle` | xvla × LIBERO / ManiSkill: 20D EE6D → 7D (pos + axis-angle + grip) |
| `ee6d_to_simpler_abs_euler` | xvla × SimplerEnv WidowX: rot6d→euler + offset + grip mapping |
| `ee6d_to_calvin_abs` | xvla × CALVIN official absolute-pose protocol |
| `ee6d_to_euler` / `ee6d_to_quat` | Other EE variants |
| `pi05_aloha_robotwin` | pi05 × RoboTwin joint decoder (stateful; via bridge) |
| `ee6d_robotwin_ee_dual` | xvla × RoboTwin dual-arm ee decoder (via bridge) |

- pi05: `action_encoding == adapter.action_space` (e.g. `axis_angle`) → empty key → passthrough.
- xvla: `action_encoding: ee6d` × each benchmark's `action_space` → auto-select the key from the table above.

If the new model's output is inconsistent with the environment's native action space, register the corresponding decoder under `action_decoders/`
(`@register_action_decoder("<encoding>_to_<space>")`);
**do not** write environment special-casing into the training-side `predict_action`.

### 3.2 `benchmark.action_bridge` (RoboTwin only)

Implementation: `bridges/robotwin_policy.py` (`_BRIDGE_WIRING` maps a bridge name to
`(model_type, payload-builder state_encoding, decoder key)`, assembling the shared
adapter → PayloadBuilder → PolicyClient → ActionDecoder four-component chain).

| bridge | Use |
|---|---|
| `pi05_aloha_14d` | **pi05 RoboTwin official protocol** (`Pi05PayloadBuilder(state_encoding="aloha_pi")` + `pi05_aloha_robotwin` decoder: adapt_to_pi + delta→abs, stateful) |
| `ee6d_dual` | **xvla RoboTwin official protocol** (`XVLAPayloadBuilder(state_encoding="ee6d_dual")` + `ee6d_robotwin_ee_dual` decoder; 20D EE, three views, `action_type='ee'`) |

Protocol logic is placed in named bridge modes, avoiding changes to the model's default behavior that would affect other benchmarks.

---

## 4. state / action semantics

### 4.1 Action space and dimensions (action_dim / action_mode)

The action dimension **changes with the benchmark protocol** and cannot simply be hard-coded to LIBERO.

- **pi05 + LIBERO / single-arm EE class:** outputs 7 dims (position delta 3, axis-angle delta 3, gripper 1),
  consistent with the LIBERO env action space, and can be issued directly.
- **pi05 + RoboTwin:** joint protocol, 14 dims, `model.action_dim: 14`, `action_horizon: 32`,
  paired with `benchmark.action_bridge: pi05_aloha_14d` (see §3.2), **not** 7 dims.
- **xvla:** model side EE6D 20 dims (`model.action_mode: ee6d`, `real_action_dim: 20`,
  PayloadBuilder capability `action_encoding: ee6d`).
  - Single-arm (LIBERO / SimplerEnv / CALVIN): the eval-side ActionDecoder converts to the env action space
    (e.g. LIBERO 7 dims); the decoder key is auto-assembled from `{action_encoding}_to_{action_space}`.
  - Dual-arm RoboTwin: `action_bridge: ee6d_dual` (the bridge internally assembles the
    `ee6d_robotwin_ee_dual` decoder), executed via EE poses through `take_action(..., action_type='ee')`.

Points to confirm: the total number of model output dimensions, the semantic layout of each dimension (position / rotation / gripper), the rotation representation
(axis-angle, 6D rotation, quaternion), and the target environment's control interface (joint vs EE).

### 4.2 Control mode (absolute-pose control vs delta control)

- **pi05 + LIBERO:** delta; robosuite OSC defaults to `use_delta=True`, no extra setting needed.
- **pi05 + RoboTwin (`pi05_aloha_14d`):** model-side delta joints → converted to absolute joints inside the bridge before being issued
  (openpi Aloha: `adapt_to_pi` decode state → inference → delta→abs → `adapt_to_pi` encode).
- **xvla + LIBERO:** absolute EEF pose. When configured with `benchmark.control_mode: absolute`
  (or the default `auto` with a non-empty decoder key auto-assembled), the runner sets OSC to `use_delta=False`.
- **xvla + SimplerEnv WidowX:** absolute EE, depending on SimplerEnv registering
  the `arm_pd_ee_target_base_pose_*` controller (see `examples/embodied/xvla/eval/SIMPLERENV_PATCH_en.md`).
- **xvla + RoboTwin:** EE absolute-pose path (`ee6d_dual`), not the joint delta path.

It should be emphasized that an absolute pose cannot be crudely turned into a delta by "linearly subtracting" the current pose:
axis-angle rotation does not satisfy linear subtraction, and the delta mode often has action scaling. A wrong control mode is
the primary reason xvla initially had a 0 success rate on LIBERO.

### 4.3 Proprioceptive input (state_encoding and data layout)

The adapter **no longer encodes** proprio: it only exposes the raw EE / joint fields in `canonical["state_raw"]`
(`eef_pos` / `eef_quat` / `ee_ori_mat` / `joint` / `endpose` / `robot_obs`, etc.);
the encoding is done by **each model's PayloadBuilder** according to `model.state_encoding`.

- **pi05 + LIBERO / CALVIN / SimplerEnv:** `state_encoding: ""` (no state kwarg emitted).
- **pi05 + ManiSkill:** `state_encoding: passthrough` (directly sends `state_raw["joint"]`, 8D Panda qpos;
  when the raw qpos is 9D, take the 7 arm joints + the mean of the 2 finger joints → 8D).
- **pi05 + RoboTwin:** `state_encoding: aloha_pi` (the PayloadBuilder does the openpi adapt_to_pi
  decode internally, 14D joints → pi space).
- **xvla:** `state_encoding: ee6d`, 20-dim input, isomorphic to the action space.
  The 6D rotation must be column-major `[R00,R10,R20, R01,R11,R21]`
  (the first two columns of the rotation matrix concatenated, aligning with X-VLA `Mat_to_Rotate6D`).
  Row-major `mat[:, :2].flatten()` will cause a wrong input distribution — this is the second reason xvla had a
  0 initial success rate on LIBERO. Other xvla encodings: `ee6d_calvin` (tcp_pos + euler→rot6d interleaved),
  `ee6d_widowx` (SimplerEnv, stateful closed-loop feedback), `ee6d_dual` (RoboTwin dual-arm, stateful).

In addition, the source of the proprio semantics must be confirmed: the original X-VLA client feeds back the previous step's predicted action;
the stateful encodings (`ee6d_widowx` / `ee6d_calvin`) already implement this closed-loop feedback via the PayloadBuilder's
`update_from_response`, while `ee6d_dual` feeds back the previous step's decoded ee action via `note_env_action`.
Single-arm LIBERO uses the environment's measured state and has been validated as feasible; if a new model is sensitive, an ablation is needed.

Boundary: `canonical["state_raw"]` holds the raw fields (for the PayloadBuilder to encode + for trace/debug);
the `state` kwarg produced by the PayloadBuilder's `build()` enters `predict_action(state=...)` via RPC.
Do not pass a nested dict directly as `state` to `predict_action` (unless the model explicitly declares that layout);
prefer passing a flat `float32` vector aligned with the training `observation.state`.

### 4.4 Normalization approach (external statistics file vs model-internal normalization)

- **pi05:** depends on an external `dataset_statistics.json` (q01/q99 denormalization);
  `server.dataset_statistics_path` must be configured (for RoboTwin,
  `examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json` can be used).
- **xvla:** normalization is inside the model's action space (the EE6DActionSpace in `action_hub.py`);
  configure `dataset_statistics_path: ""`.

Points to confirm: denormalization is performed in only one place (the model-internal `predict_action`),
and the generic policy does **not** do unnorm (see the ownership convention in §2.7 for details).

### 4.5 Model-specific request fields (domain_id, etc.)

Model-specific fields are declared by the **PayloadBuilder** and injected into the `predict_action` kwargs; the YAML writes them under the `model:` section.

- **pi05:** no `domain_id`; normalization uniformly goes through dataset_statistics. `unnorm_key` is an optional
  PayloadBuilder-internal field (`model.unnorm_key`), emitted only when the model needs to select statistics by key.
- **xvla:** a multi-domain model, configured with `model.domain_id` (when not given explicitly, the PayloadBuilder
  auto-picks it from `DEFAULT_DOMAIN_ID_MAP` by `benchmark_name`). The PayloadBuilder writes it into the RPC payload,
  and the factory wrapper converts the int to a `LongTensor`. A misconfiguration usually produces **no error**, but the action distribution is wrong.

The xvla domain_id values already used in the repo (defer to the official eval / production YAML, do not fabricate):

| Benchmark | domain_id |
|---|---|
| SimplerEnv WidowX / Bridge | **0** |
| CALVIN | 2 |
| LIBERO | 3 |
| RoboTwin2 | 6 |
| VLABench (if integrated) | 8 |

The same applies to a new model's task embedding / domain embedding / special prompt, etc.:
`model:` YAML → PayloadBuilder → RPC payload → factory → model, end to end.
If the PayloadBuilder emits a kwarg the model signature does not accept, `_filter_supported_kwargs` drops it with a
WARNING (see §2.4); use this to diagnose misconfigurations.

### 4.6 Action chunk length and execution policy

- **pi05 + LIBERO:** `action_horizon` 50, consumed by the runner per config.
- **pi05 + RoboTwin:** `action_horizon: 32` (consistent with openpi RoboTwin training).
- **xvla:** the model outputs 30 steps at once; the original client executes only the first 10 steps before replanning;
  controlled by `server.chunk_execute_steps` (default 0 → truncate to 10; can be set to N or -1 to disable truncation).
  `GenericPredictActionPolicy` is responsible for chunk caching / shape checking / dim truncation.

Points to confirm: the open-loop execution length at training time; consuming a chunk that is too long or too short will significantly affect the closed loop.

### 4.7 Number and order of image views

Rules of `payload_builders/pi05.py::_pack_images` (the xvla PayloadBuilder reuses the same function) (**not** a fixed 2 views):

1. There must be a `primary` or `head`;
2. If `left` and `right` both exist → **3 views** `[primary, left, right]` (RoboTwin / X-VLA official);
3. Otherwise append at most one more `wrist` (or a standalone left/right) as the 2nd view.

The adapter declares its camera set with the `cameras` class attribute (e.g. LIBERO `("primary", "wrist")`),
and the PayloadBuilder packs the model's expected view list from `canonical["images"]` (a per-camera dict) per the rules above.

- **pi05 × LIBERO:** usually 2 views (agentview + wrist).
- **pi05 / xvla × RoboTwin:** 3 views; the model side must support a **dynamic** `num_images = len(images[0])`,
  and **must not** hardcode `num_images=2` or `3` (otherwise it breaks other benchmarks).
- **xvla × SimplerEnv:** official single view (third person).

Points to confirm: the number, order, and resolution of cameras at training time, and whether they are flipped
(LIBERO agentview vertical flip is handled uniformly by the adapter).

Views are dynamically packed by the PayloadBuilder from the obs; do not rely on fabricated YAML fields to control the number of views.

---

## 5. Engineering parameters

- **Load timeout (`start_timeout_sec`):** xvla cold start can be >900s, and task-success configs commonly use 2400;
  900 is usually enough for pi05; a `random_init` chain smoke can be shorter.
- **processor / tokenizer:** pi05 needs an external paligemma (`tokenizer_path`);
  xvla's processor is in the checkpoint directory (`processor_path` / `tokenizer_path` are often the same directory).
- **`server.random_init`:** chain smoke when there are no weights; paired with an empty `ckpt_path`.
- **Ports:** each run uses an independent `port` / `health_port` to avoid chunk-cache cross-contamination.
- **GPU:** usually one card per policy server; running eval tasks serially is more reliable.
- **Environment separation:** the orchestrator uses the **benchmark conda**; `server.python` uses the **model server** environment.
- **SAPIEN (SimplerEnv / RoboTwin / ManiSkill):** besides `nvidia-smi`, use `vulkaninfo`
  to confirm `deviceName=NVIDIA` (not llvmpipe); set `LD_LIBRARY_PATH` and `VK_ICD_FILENAMES`.

Eval parameters (max_steps / num_steps_wait, etc.) must be configured according to the **original eval protocol + the current smoke intent**, not mixed:

- **pi05 × LIBERO:** smoke commonly uses `max_steps: 300`; full-scale long-horizon suites such as libero_10
  are recommended to be higher (e.g. 520). A `max_steps` that is too small will judge in-progress episodes as failures.
- **xvla × LIBERO:** original horizon 800, `num_steps_wait: 10`; smoke can be 1 task × 1 ep,
  but keeping 800 is still recommended so long tasks are not truncated.
- **xvla × SimplerEnv WidowX:** official `max_steps: 1200` (task-success config).
- **Chain smoke (random_init):** deliberately short step counts (e.g. 20–30), only to prove RPC, not counted as a result.

runner semantics: when `benchmark.max_steps > 0`, it takes precedence over the suite default (not bounded by its cap).

---

## 6. pi05 vs xvla configuration comparison table

### 6.1 LIBERO (primary semantic acceptance path)

| Config item | pi05 | xvla |
|---|---|---|
| model_type | pi05 | xvla |
| action_dim (env side) | 7 | 7 (produced by decoder) |
| real_action_dim (model side) | 7 (padded to 32) | 20 (EE6D) |
| action_mode | — | ee6d |
| action_encoding (PayloadBuilder) | axis_angle | ee6d |
| control mode | delta (`control_mode: auto`→delta) | absolute (`control_mode: absolute` or auto+non-empty decoder key) |
| action decoder (auto-assembled) | empty key→Identity passthrough | ee6d_to_axis_angle |
| chunk_execute_steps | — | 10 (`server.chunk_execute_steps`) |
| state_encoding (PayloadBuilder) | "" (no state emitted) | ee6d (20 dims, rot6d column-major) |
| dataset_statistics_path | required | "" |
| domain_id (`model:` section) | — | 3 |
| chunk execution length | 50 | 10 (factory truncation) |
| max_steps (protocol reference) | smoke 300; higher for long suites | 800 |
| num_steps_wait | 10 | 10 |
| image views | 2 (agentview+wrist) | 2 (same LIBERO cameras) |
| tokenizer/processor | external paligemma | checkpoint built-in |
| start_timeout_sec | 900 | 2400 |

### 6.2 Other validated / chain combinations (summary)

| Combination | Key config | Status |
|---|---|---|
| pi05 × RoboTwin | `action_dim: 14`, `action_horizon: 32`, `action_bridge: pi05_aloha_14d`, stats JSON | task-success |
| xvla × RoboTwin | `domain_id: 6`, `action_bridge: ee6d_dual`, 3 views | task-success |
| xvla × SimplerEnv WidowX | `domain_id: 0`, `ee6d_to_simpler_abs_euler`, abs EE + patch, `max_steps: 1200` | task-success |
| GR00T-N1.6 × LIBERO | `embodiment_tag: libero_panda`, `state_encoding: libero_ee_euler`, `action_encoding: axis_angle`, `control_mode: delta` | task-success (object 5/5) |
| GR00T-N1.6 × SimplerEnv WidowX | `embodiment_tag: oxe_widowx`, `state_encoding: simpler_widowx`, `action_encoding: simpler_abs_euler`, `rotation_mode: axis_angle`, `prepackaged_config: true`, `chunk_execute_steps: 4` | task-success (eggplant 20/20) |
| GR00T-N1.6 × SimplerEnv WidowX drawer | `widowx_open_drawer` / `widowx_close_drawer` (ported env), same config as above | env+config aligned to official & verified; task-success NOT reproduced (open ~0, close ~12% vs official 95/73) — model inference residual, not config (see `examples/embodied/groot_n1_6/eval/DRAWER_调查报告.md`) |
| pi05 × SimplerEnv / CALVIN / ManiSkill | `random_init: true` (no domain weights) | chain smoke only |
| xvla × CALVIN / ManiSkill | `random_init: true` (official Calvin weights not wired) | chain smoke only |

#### GR00T-N1.6 (`Gr00tN1d6`) × LIBERO — integration notes

- Factory `factories/groot_n1_6_factory.py` (`@register_factory("gr00tn1d6")`); PayloadBuilder `payload_builders/groot_n1_6.py` (`@register_payload_builder("gr00tn1d6")`). Registry keys are lowercased (`EvalServerArgs.model_type` is lowercased); the model-class registry key stays `Gr00tN1d6`.
- Multi-embodiment model: the factory calls `GrootN1d6Policy.configure_predict_action(embodiment_tag=...)` from the new `server.embodiment_tag` field (LIBERO → `libero_panda`, projector id 2).
- State: adapter emits `state_raw.gripper_qpos` (native 2-DoF Panda finger qpos); PayloadBuilder `libero_ee_euler` builds the 8D `libero_panda` raw state `[x,y,z, roll,pitch,yaw (from eef_quat, xyz-euler), finger0, finger1]`. The processor normalizes it inside `predict_action` from the checkpoint's `experiment_cfg/dataset_statistics.json`.
- Action: model outputs a 7D delta action `[dx,dy,dz, d(axis-angle)×3, gripper∈[0,1]]`; `action_encoding: axis_angle` composes an IdentityDecoder vs LIBERO's `axis_angle` space, and `benchmark.control_mode: delta` drives robosuite OSC delta. Gripper uses the adapter's default `binarize_gripper_open` (verified correct for GR00T on object).
- Eagle backbone loads offline from the local processor dir (its `config.json` is the Eagle3VL config) via the repo-local builder — set `CUDA_GRAPH_IMPL=local` (baked into the run scripts) so `_use_graph_safe_eagle()` avoids the HF remote-code path. Needs `flash_attn` in the server env, or set `model.use_flash_attention: false` for sdpa.

#### GR00T-N1.6 (`Gr00tN1d6`) × SimplerEnv WidowX (Bridge)

See `examples/embodied/groot_n1_6/eval/SIMPLERENV_en.md` for the full reproduction guide. Key points and pitfalls that cost time:
- Embodiment `oxe_widowx` was hand-added to `groot_n1_6/transforms/utils.py`. Its **action normalization must match the checkpoint's `processor_config.json`**: `mean_std_embedding_keys=[x,y,z,roll,pitch,yaw]` (pos+rotation mean-std, gripper min-max). Deriving it from `statistics.json` ranges (defaulting to min-max) blows the ±2π euler action up — and LIBERO's small ranges hide the bug.
- Proprio `simpler_widowx`: our ManiSkill obs has no `agent.eef_pos`, so reconstruct ee pose from `base_pose⁻¹ · tcp_pose`, then `rpy = mat2euler(quat2mat(quat) @ default_rot.T)` with `default_rot=[[0,0,1],[0,1,0],[-1,0,0]]` (official `WidowXBridgeEnv`). Action rotation is passed straight to the delta controller (`rotation_mode: axis_angle`, no euler→axis-angle).
- **Use the official eval protocol**: `benchmark.prepackaged_config: true` (env applies visual-matching config + per-episode randomization). Hand-set scene/overlay/init scored 36% vs 100% prepackaged. Report over ≥20 episodes (stochastic sampling); use `chunk_execute_steps: 4` (official n_action_steps).
- WidowX drawer tasks (`widowx_open_drawer`/`widowx_close_drawer`) are ported from NVIDIA's pinned fork (`open_small_drawer_in_scene.py` + `small_drawer.urdf` + `bridge_small_drawer.png`); base env already supports `dummy_drawer` + `widowx`. The ported drawer env must also override `_get_obs_extra` to emit `tcp_pose` (like grasp/pick envs) — otherwise `obs.extra` is empty, the adapter's proprio silently becomes a zero state, and the precision drawer fails while zero-tolerant eggplant still passes. Use the official client's `--max-episode-steps 300`, not the drawer env's registered `120`.

---

## 7. Integration acceptance workflow + status of configurable items

### 7.1 Integration acceptance workflow

1. **Action value check:** record the model output for the first few steps and the current pose, confirm the magnitude, coordinate frame,
   and gripper value range (±1 or [0,1]) before running a long eval.
2. **Compare against the original inference:** the official eval client (such as X-VLA's libero / simpler / robotwin client,
   openpi RoboTwin) is the behavior baseline; verify proprio, controller, chunk, and view order.
3. **Scale up in layers:** local interface validation (see §2.8) → mock/RPC → random-init chain →
   real-weight short episode → trustworthy task-success / full-scale.
4. **Shared-code regression:** after modifying shared logic in the runner, adapter, PayloadBuilder, ActionDecoder, bridge, or factory,
   you must regress with an already-succeeding combination (pi05×LIBERO small smoke recommended).
   - Last full regression (2026-07-31, after the GR00T-N1.6 shared-code changes: `adapters/simplerenv.py` tcp_pose/eef_pos/drawer maps, `adapters/libero.py` gripper_qpos, `simplerenv_runner` prepackaged_config, `EvalServerArgs` new fields): **10/10 pass** — task-success 5/5 (pi05×LIBERO, xvla×LIBERO, pi05×RoboTwin [4/4 on recheck; a single 1-episode miss first], xvla×RoboTwin, xvla×SimplerEnv) and chain-smoke 5/5 (pi05×SimplerEnv, pi05/xvla×CALVIN, pi05/xvla×ManiSkill). Confirms no regression to pi05/xvla.
5. **Docs and YAML header comments:** distinguish mock / random-init / real checkpoint / task-success;
   write the open weight URL and internal path, and do not report a chain smoke as a result.

### 7.2 Status of configurable items (implemented and to be implemented)

Design principle: anything belonging to the model's training protocol (action space, control mode, proprioception format,
normalization, chunk policy, camera views, domain ID) is implemented as YAML-configurable (with defaults);
anything belonging to the generic pipeline (RPC, chunk caching, latency, result recording) is transparent to the model.

#### Implemented configurable items

The following items have landed in the multi-benchmark integration of pi05 / xvla (LIBERO xvla object full-scale is about 94%;
for RoboTwin / SimplerEnv, etc. see the README status table).

| Config item | Implementation location | Description | Value example |
|---|---|---|---|
| `model.action_encoding` × `adapter.action_space` | `orchestrator/config.py::resolve_action_decoder_key` + `action_decoders/` `ACTION_DECODER_REGISTRY` | Auto-assemble the `{enc}_to_{space}` decoder key; same source and same space returns an empty key → Identity passthrough | `axis_angle`×`axis_angle`→Identity / `ee6d`×`axis_angle`→`ee6d_to_axis_angle` |
| `benchmark.control_mode` | LIBERO runner `_resolve_libero_use_delta` | OSC absolute/delta; **LIBERO only** (unrelated to the SimplerEnv/ManiSkill simulation `control_mode` string) | `auto` (default) / `absolute` / `delta` |
| `server.chunk_execute_steps` | xvla factory wrapper | Open-loop execution length truncation | `0`→default 10; `N>0`→truncate to N; `<0`→no truncation |
| `benchmark.action_bridge` | `bridges/robotwin_policy.py` `_BRIDGE_WIRING` | RoboTwin protocol mode (maps to PayloadBuilder + decoder) | `pi05_aloha_14d` / `ee6d_dual` |
| `model.state_encoding` | `payload_builders/<model>.py` | proprio encoding (the adapter only emits `state_raw`) | `""` / `passthrough` / `aloha_pi` / `ee6d` / `ee6d_calvin` / `ee6d_widowx` / `ee6d_dual` |
| `model.domain_id` | PayloadBuilder → factory wrap | Model-specific field passthrough | empty (auto-mapped) / `0`/`2`/`3`/`6` |
| `server.dataset_statistics_path` | GenericPredictActionPolicy | External stats vs model-internal normalization | path / `""` |
| `server.random_init` | factory / server | Chain smoke with no weights | `false` / `true` |
| `benchmark.max_steps` | runner | An explicit value takes precedence over the suite default | 300 / 800 / 1200 / short smoke |
| `benchmark.num_steps_wait` | runner | Environment stabilization wait | 10 |
| `benchmark.continuous_gripper` | adapter | Gripper continuous or binary | true |
| `benchmark.task_ids` | runner `run_batch` | Specify a task subset | empty / list |
| `model:` structural fields | each ModelConfig | Only write dataclass-declared fields (pi05: `action_dim`…; xvla: `action_mode`/`real_action_dim`…) | — |
| `server.start_timeout_sec` | server management | Load timeout | 900 / 2400 |
| `server.port` / `health_port` | YAML | Independent ports, prevent cache contamination | independent per run |
| view packing | `payload_builders/pi05.py::_pack_images` | 1–3 view dynamic packing | see §4.7 |

`control_mode` (LIBERO) semantics:

- `auto` (default, compatible with old YAML): a non-empty decoder key assembled → absolute (`use_delta=False`), otherwise delta.
- `absolute`: force absolute EE (the xvla LIBERO official config can write this out explicitly).
- `delta`: force delta (even if a decoder key is assembled).

`chunk_execute_steps`: written in the `server:` section, passed to the factory by `EvalServerArgs`; when not written, xvla still truncates to 10.

#### Configurable items to be implemented / still improvable

The following are still fixed or semi-fixed implementations; parameterize them when a new model hits them:

1. **Proprioception source** (`measured | predicted`).
   Status: mostly uses the environment's measured state; the original X-VLA uses predicted feedback, so an ablation switch should be retained.
2. **Gripper binarization threshold and mapping.**
   Status: the threshold is hard-coded in each decoder key per the official protocol; for a new protocol, prefer adding a decoder in `action_decoders/` rather than a global threshold.
3. **Further YAML-ization of the view policy.**
   Status: 3 views are already auto-packed by the PayloadBuilder (`_pack_images`) when left+right both exist;
   if there are non-head/left/right names or 4+ views in the future, extend the config then.

---

## 8. Related docs

- Integration operation skill: `skills/vla-model-eval-adapter/SKILL.md` (workflow, official config alignment,
  YAML layout, eval-only boundary).
- User entry and status table: `README.md`, `user_guide_en.md`, `benchmark_envs.md`.
- This document focuses on the **model semantics checklist**, the `predict_action` contract, and the pi05/xvla comparison; it does not re-paste the full command line.
