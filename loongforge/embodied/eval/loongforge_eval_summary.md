# LoongForge-VLA 离线评测模块说明

## 1. 模块定位

`eval` 是 LoongForge-VLA 的独立离线评测模块，用于在不修改 LoongForge 主框架源码的前提下，对不同 VLA 模型和 benchmark 进行统一评测。

模块将评测拆成两个进程：

- benchmark client：负责环境 reset/step、observation/action adapter、结果记录。
- model policy server：负责模型加载、推理、action 后处理。

两者通过 WebSocket + msgpack-numpy RPC 通信。用户侧只需要通过 YAML 启动：

```bash
python -m loongforge.embodied.eval.orchestrator.run --config <config.yaml>
```

YAML 中：

- `benchmark.name` 选择 benchmark，例如 `libero`、`calvin`、`simplerenv`、`robotwin`、`maniskill`。
- `model.backend` 选择模型后端，例如 `loongforge`、`mock`。
- `server.python` 指定模型 server 使用的 Python 环境。
- `run.output_dir` 指定评测结果基准目录；统一入口默认会为每次运行创建时间戳子目录，避免复用旧结果。

## 2. 整体架构

```text
YAML config
  -> loongforge.embodied.eval.orchestrator.run
    -> benchmark runner
      -> benchmark observation adapter
        -> unified observation schema
          -> WebSocket/msgpack-numpy RPC
            -> model policy server
              -> model inference
              -> action postprocess / unnormalization
          <- unified action schema
      -> benchmark action adapter
      -> environment step
      -> results / artifacts / logs
```

核心目录：

```text
eval/
  configs/                 # 按模型和 benchmark 组织的 YAML 配置
  reports/                 # 运行时生成的评测结果，不随代码提交
  adapters/                # benchmark 侧 observation/action adapter
  orchestrator/            # YAML dispatcher、server manager、benchmark runners 和扩展调度/报告工具
  protocol/                # 统一 observation/action/result schema
  servers/                 # 模型 policy server 和模型适配逻辑
  transport/               # WebSocket RPC 通信
```

## 3. LoongForge 适配逻辑

LoongForge 相关逻辑集中在 `loongforge/embodied/eval/servers`，不修改 LoongForge 主框架训练/模型源码。

当前 pi05 适配主要包括：

- 使用独立 LoongForge Python 环境启动 policy server。
- 根据 YAML 构造 pi05 policy。
- 支持从 checkpoint 加载权重。
- 支持 `model.random_init: true`，用于不加载权重的链路验证。
- server 启动后、health 上线前执行一次 dummy `_warmup_model()` 调用，确保所有延迟 import 在首个 episode 请求前完成，避免循环导入竞态。
- 将统一 observation schema 转换为 pi05 输入。
- 执行 pi05 inference，返回 action chunk。
- pi05 action 反归一化由模型 `predict_action()` 内部负责，eval 侧不再额外执行。

pi05 action 反归一化公式（模型内部实现，仅供参考）：

```text
norm = 2 * (x - q01) / (q99 - q01) - 1
x = (norm + 1) / 2 * (q99 - q01) + q01
```

其中 `q01` 和 `q99` 来自 YAML 中的 `model.dataset_statistics_path`。

## 4. 配置和报告规范

配置和报告都按模型、benchmark、单次运行组织：

```text
examples/embodied/<model>/eval/configs/
  <benchmark>/
    <run_name>.yaml

loongforge/embodied/eval/reports/
  <model>/
    <benchmark>/
      <yyyymmdd_hhmmss>_<run_name>/
        policy_server.log
        results.jsonl
        summary.csv
        suite_summary.csv
        artifacts/
          ... benchmark-specific trace / replay / video / official logs ...
```

示例：

```text
examples/embodied/pi05/eval/configs/libero/smoke_steps10.yaml
loongforge/embodied/eval/reports/pi05/libero/<timestamp>_smoke_steps10/

examples/embodied/pi05/eval/configs/simplerenv/eggplant_300step.yaml
loongforge/embodied/eval/reports/pi05/simplerenv/<timestamp>_eggplant_300step/

examples/embodied/pi05/eval/configs/robotwin/random_init_5step.yaml
loongforge/embodied/eval/reports/pi05/robotwin/<timestamp>_random_init_5step/

examples/embodied/pi05/eval/configs/maniskill/pick_cube_20step.yaml
loongforge/embodied/eval/reports/pi05/maniskill/<timestamp>_pick_cube_20step/
```

`policy_server.log` 保留在 run 根目录。benchmark 相关产物统一放到 `artifacts/` 下，内部层级由 runner 按 benchmark 特性组织。RoboTwin 运行时生成的 `worker_files/` 不作为最终报告保留，只保留归档后的 official log、deploy config、`_result.txt`、bridge `trace.json` 和可选 `mp4` 视频。

## 5. 接入新 benchmark

新增 benchmark 时应复用现有协议和模型 server，不把模型逻辑写进 benchmark runner。

推荐步骤：

1. 在 orchestrator 中注册新的 `benchmark.name`。
2. 新增 benchmark runner，负责环境创建、reset/step、结果记录。
3. 新增或复用 benchmark adapter，将原始 observation 转成统一 observation schema。
4. 将 policy server 返回的统一 action schema 转成 benchmark 环境需要的 action。
5. 将输出统一落到 `run.output_dir`。
6. 按规范新增 YAML：`configs/<model>/<benchmark>/<run_name>.yaml`。
7. 按规范输出报告：`reports/<model>/<benchmark>/<run_name>/`。
8. 先用 mock 或 random-init 做链路 smoke，再用真实 checkpoint 跑短 rollout，最后扩展到正式评测。

接入时需要特别确认：

- benchmark 需要的 action 维度。
- observation 中图像、状态、语言指令的字段含义。
- action 坐标系、gripper 语义和控制频率。
- 是否需要 benchmark 原生评测脚本或 worker-local 文件。
- 是否能生成标准 `results.jsonl`、`summary.csv`、`suite_summary.csv`；如果 benchmark 原生没有，应尽量在 runner 中补齐。

## 6. pi05 评测示例

当前 pi05 已按统一结构接入 LIBERO、CALVIN、SimplerEnv、RoboTwin 和 ManiSkill 五类 benchmark 入口；其中 CALVIN、SimplerEnv、RoboTwin 和 ManiSkill 的当前结果主要作为链路 smoke/debug，正式成绩依赖对应 benchmark/domain 的 checkpoint 和 dataset statistics。

### LIBERO

配置：

```text
examples/embodied/pi05/eval/configs/libero/smoke_steps10.yaml
```

报告：

```text
reports/pi05/libero/<timestamp>_smoke_steps10/
```

结果：

- 评测链路跑通，exit code 0。
- `libero10_smoke_internal.yaml` 在统一 `predict_action()` 重构后完成 task 0，2026-07-07 全量 smoke 中使用真实 PI05 权重，结果为 `success: 1`、`success_rate: 1.0`、`steps: 267`。
- 第一次重构 smoke 暴露了 benchmark-native dict state 不能直接传给 pi05 的问题；现在由 adapter/payload 层只向模型转发 `model_state`，最新复测未再出现 `must be real number, not dict`。
- 完整 replay/trace 产物在历史成功 smoke 中验证过；2026-07-07 全量 smoke 为节省空间关闭了 `save_replay` 和 `save_trace`，读完结果后已删除临时产物。

### CALVIN

配置：

```text
examples/embodied/pi05/eval/configs/calvin/smoke.yaml
```

结果状态：

- runner、adapter、YAML 入口和启动脚本已接入。
- CALVIN 是 7D Franka 长程 sequence 评测，一条 sequence 连续执行 5 个 subtask。
- 评测需要原始 CALVIN 格式数据和配置资产，包括 `validation/.hydra/merged_config.yaml`、`calvin_models/conf` 和 `eval_sequences.json`。
- 内部 debug dataset 已完成 30-step smoke：env reset、policy RPC、action chunk、env step 和 summary 输出均已打通。2026-07-07 全量 smoke 使用 random-init PI05 重跑，server metadata 确认为 `random_init: true`、`ckpt_path: random_init://pi05`，结果 exit code 0、`avg_length: 0.0`，仅作链路验证。

### SimplerEnv

配置：

```text
examples/embodied/pi05/eval/configs/simplerenv/eggplant_300step.yaml
```

报告：

```text
reports/pi05/simplerenv/<timestamp>_eggplant_300step/
```

结果状态：

- Bridge runner、Vulkan/SAPIEN headless runtime、reset/step loop、trace 和 GIF 输出已打通。
- 已配置任务包括 `widowx_put_eggplant_in_basket`、`widowx_carrot_on_plate`、`widowx_stack_cube` 和 `widowx_spoon_on_towel`。
- `widowx_carrot_on_plate` 60-step smoke 已完成。2026-07-07 全量 smoke 使用 random-init PI05 重跑，exit code 0、`success: 0`，符合随机初始化链路验证预期；为节省空间关闭 replay 和 trace，读完结果后已删除临时产物。
- 固定小尺度 action sanity 已验证 WidowX controller 能移动；大尺度动作会导致 IK 失败回退，表现为机械臂不动。
- 当前内部 pi05 SimplerEnv 配置使用 LIBERO 域 checkpoint 或空 `dataset_statistics_path`，只能作为链路 smoke/debug，不作为可信 benchmark score。
- 正式 SimplerEnv 评测需要 Bridge/WidowX/SimplerEnv 对应 pi05 checkpoint 和匹配的 `dataset_statistics.json`。

### RoboTwin

配置：

```text
examples/embodied/pi05/eval/configs/robotwin/random_init_5step.yaml
```

报告：

```text
reports/pi05/robotwin/<timestamp>_random_init_5step/
```

结果：

- official runner 链路跑通，exit code 0。
- 使用 random-init 14D pi05，不加载真实 checkpoint。
- 2026-07-07 全量 smoke 执行 5-step official episode，return code 0，`success: 0`，符合随机初始化预期。
- official log 中可能出现 SAPIEN/Vulkan warning 或 `Render Error` 文本；本轮进程返回 0，并产出 `results.jsonl`、`summary.csv`、`suite_summary.csv` 和 official log。为节省空间，读完结果后已删除临时产物。

说明：当前真实 pi05 checkpoint 是 7D action，适用于 LIBERO 链路验证；但 CALVIN、SimplerEnv/WidowX、ManiSkill 的 action/interface/domain 与 LIBERO 不同，正式成功率评测需要对应 benchmark/domain 的 checkpoint 和匹配 dataset statistics。RoboTwin official 任务是双臂 14D action，正式成功率评测需要 14D RoboTwin pi05 checkpoint 和匹配 dataset statistics。

### ManiSkill

配置：

```text
examples/embodied/pi05/eval/configs/maniskill/pick_cube_20step_internal.yaml
```

结果：

- 2026-07-07 全量 smoke 使用 random-init PI05 跑 `PickCube-v1`。
- 完成 5-step rollout，exit code 0，`success: 0`，符合随机初始化链路验证预期。
- 运行中可能出现 SAPIEN/Vulkan ICD warning；本轮进程返回 0，并产出 `results.jsonl` 和 `summary.csv`。为节省空间，读完结果后已删除临时产物。

## 7. 验证范围

已验证的主链路：

- `run.py`、`config.py`、`server_manager.py`。
- `libero_runner.py`：pi05 LIBERO 300-step smoke，时间戳目录下 `new_records: 1`。
- `calvin_runner.py`：CALVIN 7D long-horizon runner、独立 conda env 和 30-step debug smoke 已通过；正式长程分数仍需匹配 CALVIN checkpoint/stats。
- `simplerenv_runner.py`：SimplerEnv Bridge runtime smoke、60-step carrot rollout 和固定小动作 sanity 已通过；模型分数仍为 debug/smoke。
- `robotwin_runner.py`：pi05 RoboTwin random-init 14D 5-step official runner。
- `maniskill_runner.py`：ManiSkill `PickCube-v1` random-init 5-step runner。
- `generate_report.py`、`compare_repro.py`、`archive_traces.py`：已用真实 LIBERO `results.jsonl` 做轻量 CLI 验证。

TODO：后续如果需要多任务、多 endpoint 调度，再单独引入并验证 `orchestrator/scheduler.py` 和 `orchestrator/work_queue.py`。它们不属于当前 pi05 主链路，本次同步不作为已交付功能。

## 8. 当前结论

当前评测模块已经完成 LoongForge/pi05 的基础适配：

- 不侵入 LoongForge 主框架源码。
- YAML 统一入口可用。
- LIBERO、CALVIN、SimplerEnv、RoboTwin 和 ManiSkill 五类入口已接入；其中 CALVIN、SimplerEnv、RoboTwin 和 ManiSkill 当前是链路 smoke/debug 状态，正式成绩依赖匹配 checkpoint 和 dataset statistics。
- 配置和报告已按 `model/benchmark/run_name` 统一组织。
- pi05 的 q99 action 反归一化已按 LoongForge 逻辑实现。
- 当前验证重点是评测链路可用；模型任务成功率依赖后续更合适的 checkpoint。