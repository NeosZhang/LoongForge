# Predict Action Interface

本文档说明 LoongForge embodied eval 中面向新模型接入的统一 `predict_action()` 接口，以及 eval 侧如何校验模型接口和 action 输出。

## 1. 目标

当 benchmark runner 已经接入完成后，新模型不应该再复制一份完整的 `LoongForgeXXXPolicy`。推荐方式是：

1. 模型自身实现统一 `predict_action()` 接口。
2. eval 侧新增一个轻量 model factory，负责 import、config、checkpoint、device/dtype 和 metadata。
3. server 使用 `GenericPredictActionPolicy` 复用统一 RPC、cache、latency、dataset stats、action shape 校验和 action dim 裁剪逻辑。

当前 PI05 路径已经按这个模式接入：

```text
loongforge_server.py
  -> PI05ModelFactory.build(...)
  -> GenericPredictActionPolicy(...)
  -> PI05Policy.predict_action(images, instructions, state=None, dataset_stats=None)
```

## 2. 接口定义

新模型需要暴露一个 callable：

```python
def predict_action(images, instructions, state=None, dataset_stats=None):
    ...
```

参数含义：

- `images`：batch 后的图像输入。当前 generic policy 会按 benchmark canonical image views 选择 primary/wrist 等视角，并传给模型。
- `instructions`：batch 后的语言指令列表。
- `state`：可选模型状态输入，来自 adapter 产出的 `model_state`。当前 PI05 smoke 路径默认传 `None`。
- `dataset_stats`：可选 dataset statistics，用于模型侧或 eval 侧执行归一化/反归一化相关逻辑。

注意：接口参数名保持 `state`，因为这是模型 `predict_action()` 的通用语义；adapter 侧字段叫 `model_state`，表示“准备传给模型的 state”。runner 会做如下转换：

```python
payload = {
    "images": canonical_obs["images"],
    "instruction": canonical_obs["instruction"],
    "state": canonical_obs.get("model_state"),
}
```

链路是：

```text
adapter.model_state -> RPC payload.state -> predict_action(state=...)
```

## 3. 校验位置

接口校验代码位于：

```text
loongforge/embodied/eval/servers/predict_action_interface.py
```

核心对象和函数：

```python
class PredictActionModel(Protocol):
    def predict_action(
        self,
        images: Any,
        instructions: Any,
        state: Optional[Any] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...
```

```python
def validate_predict_action_model(model: Any) -> None:
    ...
```

```python
def call_predict_action(
    model: PredictActionModel,
    images: Any,
    instructions: Any,
    state: Optional[Any],
    dataset_stats: Optional[Dict[str, Any]],
    action_dim: int,
) -> np.ndarray:
    ...
```

`GenericPredictActionPolicy` 会通过 `call_predict_action()` 调用模型，因此只要新模型走 generic policy 路径，就会自动经过接口校验和 action 输出校验。

## 4. validate_predict_action_model 检查什么

`validate_predict_action_model(model)` 会检查：

- `model.predict_action` 是否存在。
- `model.predict_action` 是否 callable。
- 是否包含必需参数：
  - `images`
  - `instructions`
- 是否可以接收 eval 侧会传入的可选 keyword：
  - `state`
  - `dataset_stats`
- 如果模型接口使用 `**kwargs`，则允许它接收这些可选参数。

错误示例：

```python
class BadModel:
    def predict_action(self, images):
        ...
```

该接口缺少 `instructions`，并且不能接收 `state` / `dataset_stats`，eval 会在调用前抛出 `TypeError`。

可接受示例：

```python
class GoodModel:
    def predict_action(self, images, instructions, state=None, dataset_stats=None):
        ...
```

也可以使用 `**kwargs` 兼容额外参数：

```python
class KwargsModel:
    def predict_action(self, images, instructions, **kwargs):
        state = kwargs.get("state")
        dataset_stats = kwargs.get("dataset_stats")
        ...
```

## 5. action 输出要求

模型 `predict_action()` 可以返回以下 shape：

```text
[D]
[H, D]
[B, H, D]
```

`call_predict_action()` 会统一规整为：

```text
[H, action_dim]
```

规整规则：

- `[D]` 会 reshape 成 `[1, D]`。
- `[H, D]` 原样作为 action chunk。
- `[B, H, D]` 会 reshape 成 `[-1, D]`，用于当前单请求 action chunk 处理。
- 其他维度会报 `ValueError`。
- 如果输出最后一维小于 `action_dim`，会报 `ValueError`。
- 如果输出最后一维大于 `action_dim`，会裁剪到前 `action_dim` 维。

例如 benchmark 需要 7D action：

```python
actions = model.predict_action(...)
# actions shape: [50, 32]
```

则 eval 会返回：

```python
actions[:, :7]
# shape: [50, 7]
```

## 6. model_state 和 state 的关系

benchmark adapter 可以保留结构化状态：

```python
canonical_obs = {
    "state": {
        "eef_pos": [...],
        "eef_rot_axis_angle": [...],
        "gripper": ...,
        "frame": "base",
        "units": {"pos": "m", "rot": "rad"},
    },
    "model_state": None,
}
```

含义：

- `state`：benchmark/adapter/debug/trace 使用的结构化状态。
- `model_state`：真正准备传给模型 `predict_action(state=...)` 的状态。

当前 PI05 路径下，LIBERO、CALVIN、SimplerEnv、RoboTwin、ManiSkill 默认：

```python
"model_state": None
```

原因是 PI05 可以接收数值 state，但它要求该 state 与训练时的 `observation.state` 在维度、顺序、单位、坐标系、gripper 表示和 `dataset_stats["observation.state"]` 上一致。当前 benchmark adapter 的结构化 dict 不能直接当作 PI05 state 传入。

如果未来某个模型需要状态输入，adapter 应显式产出模型可接受的数值状态：

```python
canonical_obs = {
    "state": {
        "eef_pos": [0.12, -0.03, 0.45],
        "eef_rot_axis_angle": [0.01, 0.02, 0.03],
        "gripper": 0.04,
        "frame": "base",
        "units": {"pos": "m", "rot": "rad"},
    },
    "model_state": [0.12, -0.03, 0.45, 0.01, 0.02, 0.03, 0.04],
}
```

此时模型实际收到：

```python
predict_action(..., state=[0.12, -0.03, 0.45, 0.01, 0.02, 0.03, 0.04])
```

## 7. 本地接口验证示例

模型开发者可以先不启动 benchmark 或 policy server，只在模型环境中直接验证接口签名和 action 输出 shape。

下面示例用一个最小 mock model 演示验证方式：

```bash
cd /workspace/LoongForge-VLA
PYTHONPATH=/workspace/LoongForge-VLA python - <<'PY'
import numpy as np

from loongforge.embodied.eval.servers.predict_action_interface import (
    call_predict_action,
    validate_predict_action_model,
)

class MyModel:
    def predict_action(self, images, instructions, state=None, dataset_stats=None):
        batch_size = len(instructions)
        horizon = 4
        action_dim = 7
        return np.zeros((batch_size, horizon, action_dim), dtype=np.float32)

model = MyModel()
validate_predict_action_model(model)
actions = call_predict_action(
    model,
    images=[[np.zeros((224, 224, 3), dtype=np.uint8)]],
    instructions=["pick up the cube"],
    state=None,
    dataset_stats=None,
    action_dim=7,
)
print(actions.shape)
PY
```

期望输出：

```text
(4, 7)
```

接入真实模型时，将 `MyModel()` 替换成对应 model factory/loader 构造出的模型实例即可。这个检查只验证统一接口和 action shape；完整 eval 链路仍需要通过 YAML 启动 benchmark runner 和 policy server 做 smoke test。

## 8. 新模型接入检查清单

接入新模型时至少确认：

- 模型暴露 callable `predict_action(images, instructions, state=None, dataset_stats=None)`。
- `validate_predict_action_model(model)` 可以通过。
- `predict_action()` 返回 `[D]`、`[H, D]` 或 `[B, H, D]`。
- 输出最后一维至少为目标 benchmark 的 `action_dim`。
- `model.action_dim` 与 benchmark action adapter 期望一致，例如单臂 7D、RoboTwin 双臂 14D。
- 如果使用 `model_state`，其语义、维度、顺序和 `dataset_stats["observation.state"]` 必须与模型训练数据一致。
- model factory 只处理模型私有加载逻辑，不清理 benchmark-native dict state。
- benchmark-native observation/state 结构应在 adapter/payload 层处理。

## 9. 常见错误

### 缺少 predict_action

```text
TypeError: model must expose a callable predict_action(images, instructions, state, dataset_stats)
```

说明模型没有提供 eval 需要的统一入口。

### 缺少必需参数

```text
TypeError: model.predict_action is missing required parameters: ['instructions']
```

说明接口签名不满足要求。

### 不能接收 state 或 dataset_stats

```text
TypeError: model.predict_action cannot accept eval keyword parameters: ['state']
```

说明模型接口没有声明 `state`，也没有 `**kwargs`。

### action shape 不支持

```text
ValueError: model.predict_action returned unsupported action shape: (...)
```

说明模型输出不是 `[D]`、`[H, D]` 或 `[B, H, D]`。

### action_dim 不够

```text
ValueError: model.predict_action returned action dim X, expected at least Y
```

说明模型输出维度小于 benchmark 需要的 action 维度。
