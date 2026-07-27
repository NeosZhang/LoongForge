6.8. quick_start_Cosmos3.md

本文档将引导您完成 LoongForge-VLA 框架中 **Cosmos3-Nano** SFT

对应的开源框架文档 [https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md)

## 0. 资源准备
* **模型权重：**
    * cosmos3 [https://huggingface.co/nvidia/Cosmos3-Nano/tree/main](https://huggingface.co/nvidia/Cosmos3-Nano/tree/main)
    * vae [https:a//huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth?download=true)

* **Tokenizer：**[https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main)
* **数据集：**[https://huggingface.co/datasets/nvidia/Cosmos3-DROID/tree/main/success](https://huggingface.co/datasets/nvidia/Cosmos3-DROID/tree/main/success)

### 0.1 模型权重
参考 [https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md#step-2--prepare-checkpoint](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md#step-2--prepare-checkpoint) 将下载的huggingface权重转换成DCP格式。

通过参数`--pretrained-checkpoint` 接入转换后的DCP权重。

通过参数 `--init-on-meta`使权重在 FSDP wrap 之后再加载。

### 0.2 Tokenizer
通过参数`--tokenizer-path` 接入下载的tokenizer。

### 0.3 数据集
通过参数 `--dataset-path` 接入下载的数据集

同时，需要通过参数

`--dataset-format lerobot_datasets `

`--dataset-strategy cosmos3_droid` 

指定数据处理策略，具体包括画面拼接、图像增强等。

## 1. 启动训练
脚本：`examples/embodied/cosmos3/run_cosmos3_nano_droid_fsdp.sh`

### 1.1 训练参数
* 模型初始化

```bash
--model-name cosmos3_nano            # 指定模型
--distributed-strategy fsdp          # 分布式策略：FSDP2 全分片
--dtype bfloat16                     # 训练精度 bf16
--init-on-meta                       # 通过 meta device 降低模型初始化时的峰值显存
```
* 数据集

```bash
--dataset-format lerobot_datasets    # 数据集格式：LeRobot v3.0
--dataset-strategy cosmos3_droid     # Cosmos3 DROID 数据处理策略
--dataset-path $DATASET_PATH         # 数据集目录
--tokenizer-path $TOKENIZER_PATH     # Qwen3-VL tokenizer / processor 目录
--num-workers 4                      # DataLoader worker 数
```
* 训练策略

```bash
--disable-tf32                       # 关闭 TF32，避免影响模型效果）
--lr-group net.action2llm=1e-3,net.llm2action=1e-3,net.action_modality_embed=1e-3,net=2e-4
                                     # 分组学习率：动作头(action2llm/llm2action/action_modality_embed) 1e-3，其余 net 2e-4
--lr-decay-style lambda_linear       # 学习率衰减：线性
--optimizer TorchFusedAdamW          # 优化器：fused AdamW
```
