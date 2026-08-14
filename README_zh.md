<p align="right"><sub><a href="./README.md">English</a> | <b>简体中文</b></sub></p>

<div align="center">

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/logo/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/logo/banner.svg">
    <img alt="LoongForge" src="./docs/assets/images/logo/banner.svg" width="520">
  </picture>
</p>

<h4>一个统一、高性能的框架，用于训练 LLM、VLM、Diffusion 与 Embodied 模型。</h4>

<p align="center">
  <a href="https://baidu-baige.github.io/LoongForge/"><b>🌐 官网</b></a>
  &nbsp;·&nbsp;
  <a href="https://loongforge.readthedocs.io/zh-cn/latest/index.html"><b>📖 文档</b></a>
  &nbsp;·&nbsp;
  <a href="https://baidu-baige.github.io/LoongForge/blog/"><b>✍️ 博客</b></a>
  &nbsp;·&nbsp;
  <a href="#quickstart"><b>⚡ 快速开始</b></a>
  &nbsp;·&nbsp;
  <a href="#performance"><b>📊 性能表现</b></a>
  &nbsp;·&nbsp;
  <a href="#models"><b>🏛️ 支持模型</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/baidu-baige/LoongForge/issues/80"><b>💬 联系我们</b></a>
</p>

</div>

## 💡 为什么选择 LoongForge？

> 🐉 LoongForge 名字源于中国传统 **龙舟**，象征协同发力与破浪前行。

**LoongForge** 是面向 **LLM、VLM、Diffusion 与 Embodied 模型** 的统一训练框架，覆盖 **预训练（Pre-training）**、**持续预训练（Continued Pre-training）** 和 **SFT**。其核心目标是覆盖主流开源模型，并提供高效的训练性能。

在开源之前，LoongForge 的前身是 **AIAK-Training-LLM** —— 百度百舸的训练加速栈，已在 **教育**、**计算机视觉** 和 **Embodied AI** 等多家企业客户的生产训练中落地，相对客户原有方案通常带来 **30%~50% 加速**，最大规模的生产训练任务达到 **5,000+ XPU**。

<a id="architecture"></a>
## 🏗️ 架构

由于不同模型场景的训练需求各异，LoongForge 构建在多种分布式训练后端之上。其中 LLM/VLM/Diffusion 采用 **Megatron-LM**，具身模型采用 **torch 原生 DDP/FSDP** 技术栈。每条路线都经过深度性能优化，从而超越主流开源方案。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/architecture/loongforge-architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/architecture/loongforge-architecture.svg">
    <img alt="LoongForge 架构图" src="./docs/assets/images/architecture/loongforge-architecture.svg" width="100%">
  </picture>
</p>

## 🔥 最新动态

- **[2026/07]** 🤖 发布 **[LoongForge-Embodied](./loongforge/embodied)** 子系统，提供主流 **VLA** 与 **WAM** 模型的训练支持。
- **[2026/07]** ✨ 新增 **DeepSeek v4 flash / DeepSeek v4 pro** 训练支持。
- **[2026/05]** ⚡ **Wan 2.2** 训练 **加速 116%**，并新增 CP（上下文并行）与数据 packing 策略支持。
- **[2026/05]** ✨ 新增 **Kimi K2.5 / K2.6** 训练支持，并支持 **INT4 / NVFP4** PTQ 量化能力。
- **[2026/05]** 🎉 **v0.1.0** —— LoongForge 首个正式版本发布。
- **[2026/05]** 🌟 支持 **LLaVA-OneVision-2.0** 模型训练并协助其公开发布。
- **[2026/05]** 🤖 扩展 VLA 模型覆盖，新增 **GR00T N1.6**；Pi0.5 与 GR00T 训练实现 **60%+ 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-06-loongforge-groot-n16-acceleration.html)]
- **[2026/04]** 🧩 新增 **MiniMax-M2.7** 在 NVIDIA GPU 与昆仑芯 XPU 上的训练支持。
- **[2026/04]** 🚀 LoongForge 源码在 GitHub 上正式公开。[[blog](https://zhuanlan.zhihu.com/p/2031006068797600446)]
- **[2025/10]** 🌟 基于AIAK-Training-LLM（LoongForge 前身）支持 **LLaVA-OneVision-1.5** 模型训练并协助其公开发布。[[blog](https://mp.weixin.qq.com/s/1y7Br15pBpUZ-90j5OGncA)]

<a id="quickstart"></a>
## ⚡ 快速开始

完整的安装、教程与进阶使用请查阅文档 —— [English](https://loongforge.readthedocs.io/en/latest/index.html) · [中文](https://loongforge.readthedocs.io/zh-cn/latest/index.html)。

**1. 安装** —— 可使用 **预构建镜像** 或 **源码构建**：
- **NVIDIA GPU**：[安装指南](https://loongforge.readthedocs.io/zh-cn/latest/get_started/installation.html)
- **昆仑芯 XPU**：[安装指南](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/install_p800.html)

**2. 启动你的第一个训练任务** —— 根据目标硬件与模态选择教程：
- **NVIDIA GPU**：[LLM](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/quick_start_llm_pretrain.html) · [VLM](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/quick_start_vlm_pretrain.html) · [VLA & WAM](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/overview.html) · [Diffusion (WAN)](https://loongforge.readthedocs.io/zh-cn/latest/wan_tutorial/quick_start_wan_training.html)
- **昆仑芯 XPU**：[昆仑芯 XPU 教程](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/README.html)

**3. 深入探索** —— 浏览 [`configs/models/`](./configs/models) 和 [`examples/`](./examples) / [`examples_xpu/`](./examples_xpu) 下的现成启动脚本。

## ✨ 核心特性

* **🧩 灵活的多模态组合** —— 通过配置驱动的方式，将可互换的 ViT 与 LLM 组件自由组装为 VLM。
* **⚡ 异构并行** —— 针对模型不同组件（如 ViT vs LLM）独立配置 TP / DP / 重计算策略，获得最优吞吐与显存占用。 [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-heterogeneous-parallel-training.html)]
* **🔀 Encoder-Decoder 解耦训练** —— 将 ViT 与 LLM 拆分为独立任务，消除 Encoder 带来的流水线气泡。
* **⚖️ DP 负载均衡** —— 基于负载感知的数据重分发，缓解序列打包（sequence packing）不均衡问题，显著提升多节点扩展效率。 [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-dp-load-balancing.html)]
* **🚀 MoE 原生优化** —— All2All / 激活卸载 / 计算全链路重叠，在 DeepSeek-V3、Qwen3-MoE 等模型上相对上游 Megatron-LM 实现**进一步显存降低**。
* **🚦 MoE 专家负载均衡** —— 基于拓扑感知算法实现热点专家动态复制，均衡 MoE 专家并行训练中的计算负载，提升训练效率。 [[Paper](https://arxiv.org/pdf/2608.03676)]
* **🔬 自适应 FP8 训练** —— 面向 LLM 和 VLM 的端到端 FP8，支持标准 **blockwise FP8**；可选 **自适应** 模式根据 GEMM 形状与效率逐算子选择最佳精度。
* **🔧 自定义融合算子** —— 为 DSA 类模型设计的 **FusedDSA** 等融合 Kernel —— TileLang 版本已开源，高性能 CUDA 版本在百度百舸平台提供。
* **🔁 灵活的 Checkpoint 机制** —— 支持离线 **Megatron ↔ HuggingFace** 双向转换，以及在线原生 HF 加载/保存，全流程无格式壁垒。
* **🧰 丰富的流水线与数据工具** —— 开箱即用的 **Pretrain / MidTrain / SFT / LoRA** 流水线，内置数据集格式转换与序列打包能力。
* **🤖 具身模型训练** —— 面向 **VLA 与世界-动作模型（WAM）**（如 Pi0.5、GR00T N1.6、FastWAM）的独立 **torch 原生 DDP/FSDP** 子系统，与 Megatron 核心解耦，支持 **DDP / ZeRO-1 / FSDP / HSDP** 多种分布式策略。
* **🌐 异构硬件** —— 通过轻侵入式插件设计，原生支持 **NVIDIA GPU** 与 **昆仑芯 XPU**。

> 📖 深入阅读：[LLM 特性](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/features_index.html) · [VLM 特性](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/features_index.html)

<a id="performance"></a>
## 📊 性能表现

相对主流开源基线的训练加速。每一行都标注了测量所用的版本，并随各模型的演进单独刷新：

<img alt="LoongForge Benchmark Speedup" src="docs/assets/images/benchmark_speedup.png" />

<details>
<summary><b>📋 详细信息</b></summary>

<br>

| 模型 | 类型 | 对比基线 | 加速比 | 测量版本 |
|---|---|---|---|---|
| DreamZero (DROID Wan2.2-5B Full) | WAM | DreamZero | **2.67×** | master · 2026-07 |
| GR00T N1.6 | VLA | LeRobot | **2.31×** | master · 2026-07 |
| Pi0.5 | VLA | OpenPI | **2.23×** | master · 2026-07 |
| LingBot VA | WAM | LingBot-VA | **1.80×** | master · 2026-07 |
| X-VLA | VLA | X-VLA | **1.69×** | master · 2026-07 |
| DeepSeek-V3.2 Lite <sup>§</sup> | MoE + DSA | Megatron-LM | **5.04×** | v0.1.1 |
| Qwen3-VL-30B-A3B | VLM | VeOmni | **1.45×** | v0.1.1 |
| Qwen3-30B-A3B | MoE | Megatron-LM | **1.16×** | v0.1.1 |

> <sup>§</sup> 受测试台规模限制，**DeepSeek-V3.2** 在减层配置下单独验证 —— LoongForge 的 **DSA CUDA Kernel 优化** 相对 Megatron-LM 仍带来 **~5× 加速**，并可支持 **64K 序列长度**（基线在 8K 以上即 OOM）。<br>
> 数据反映测量时对应基线与 LoongForge 的实现（详见 **测量版本** 列），后续可能随实现演进而变化。<br>
</details>

## 🌟 基于 LoongForge 训练

基于 LoongForge 或其前身 AIAK-Training-LLM 训练的开源模型：

- [LLaVA-OneVision-2.0](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2) —— 新一代多模态模型，配套全新的 VideoCaption 和 Spatial 数据集。
- [Innovator-VL](https://github.com/InnovatorLM/Innovator-VL/tree/main) —— 面向高级推理的科学多模态大模型。
- [LLaVA-OneVision-1.5](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/1.5) —— 面向多模态训练民主化的全开源框架。
- [Qianfan-VL](https://github.com/baidubce/Qianfan-VL) —— 面向企业的领域增强视觉-语言模型，参数量覆盖 3B ~ 70B。

<a id="models"></a>
## 🏛️ 支持的模型

LoongForge 已支持 LLM、VLM、Diffusion 与 Embodied 等类别的广泛模型家族。点击模型名称可查看对应的训练示例；完整的使用说明请参阅[用户手册](https://loongforge.readthedocs.io/zh-cn/latest/index.html)，模型变体请参阅[模型支持矩阵](https://loongforge.readthedocs.io/zh-cn/latest/get_started/support_model.html)。

<table width="100%" style="table-layout: fixed; border-collapse: collapse;">
<colgroup>
<col width="25%">
<col width="25%">
<col width="25%">
<col width="25%">
</colgroup>
<thead align="center" valign="bottom">
<tr><th width="25%">LLM</th><th width="25%">VLM</th><th width="25%">Diffusion</th><th width="25%">Embodied</th></tr>
</thead>
<tbody valign="top">
<tr>
<td valign="top">
<ul>
<li><a href="examples/deepseek_v2/">DeepSeek-V2</a> ✅</li>
<li><a href="examples/deepseek_v3/">DeepSeek-V3/V3.2</a> ✅</li>
<li><a href="examples/deepseek_v4/">DeepSeek-V4</a> ✅</li>
<li><a href="examples/llama2/">LLaMA2</a> ✅</li>
<li><a href="examples/llama3/">LLaMA3</a> ✅</li>
<li><a href="examples/llama3.1/">LLaMA3.1</a> ✅</li>
<li><a href="examples/qwen/">Qwen</a> ✅</li>
<li><a href="examples/qwen1.5/">Qwen1.5</a> ✅</li>
<li><a href="examples/qwen2/">Qwen2</a> ✅</li>
<li><a href="examples/qwen2.5/">Qwen2.5</a> ✅</li>
<li><a href="examples/qwen3/">Qwen3</a> ✅</li>
<li><a href="examples/qwen3_next/">Qwen3-Next</a> ✅</li>
<li><a href="examples/minimax/">MiniMax-M2.1/2.5/2.7</a> ✅</li>
<li><a href="examples/mimo/">MIMO</a> ✅</li>
<li><a href="examples/glm5/">GLM-5</a> ✅</li>
<li><a href="examples/glm5.2/">GLM-5.2</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/qwen2.5_vl/">Qwen2.5-VL</a> ✅</li>
<li><a href="examples/qwen3_vl/">Qwen3-VL</a> ✅</li>
<li><a href="examples/qwen3.5/">Qwen3.5</a> ✅</li>
<li><a href="examples/qwen3.6/">Qwen3.6</a> ✅</li>
<li><a href="examples/kimi_k2.x/kimi_k2.5/">Kimi-K2.5/2.6</a> ✅</li>
<li><a href="examples/glm5.2_vit/">GLM-5.2 + Kimi-K2.6 ViT</a> ✅</li>
<li><a href="examples/ernie4.5/">ERNIE4.5-VL</a> ✅</li>
<li><a href="examples/llava_onevision_1.5/">LLaVA-OneVision-1.5</a> ✅</li>
<li><a href="examples/internvl2.5/">InternVL2.5</a> ✅</li>
<li><a href="examples/internvl3.5/">InternVL3.5</a> ✅</li>
<li><a href="examples/custom/">CustomCombinedModel Example</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/wan/">Wan2.1</a> ✅</li>
<li><a href="examples/wan/">Wan2.2</a> ✅</li>
<li><a href="examples/qwen_image/">Qwen-Image</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/embodied/pi05/">Pi0.5</a> ✅</li>
<li><a href="examples/embodied/groot_n1_6/">GR00T-N1.6</a> ✅</li>
<li><a href="examples/embodied/groot_n1_7/">GR00T-N1.7</a> ✅</li>
<li><a href="examples/embodied/xvla/">xVLA</a> ✅</li>
<li><a href="examples/embodied/fastwam/">FastWAM</a> ✅</li>
<li><a href="examples/embodied/lingbot_va/">LingBot-VA</a> ✅</li>
<li><a href="examples/embodied/cosmos3/">Cosmos3</a> ✅</li>
<li><a href="examples/embodied/dreamzero/">DreamZero</a> ✅</li>
</ul>
</td>
</tr>
</tbody>
</table>

## 📂 代码结构

<details>
<summary><b>📁 目录树</b></summary>

```
LoongForge/
├── loongforge/                   # 核心训练框架
│   ├── train/                    # 训练入口与训练器
│   │   ├── pretrain/             #   预训练（LLM、VLM）
│   │   ├── sft/                  #   SFT（LLM、VLM、InternVL、ERNIE）
│   │   └── diffusion/            #   Diffusion（WAN、Qwen-Image）
│   ├── models/                   # 统一的模型抽象层
│   │   ├── foundation/           #   LLM 主干（LLaMA、Qwen、DeepSeek、...）
│   │   ├── encoder/              #   视觉编码器（ViT、Qwen-VL、InternVL、...）
│   │   ├── omni_models/          #   多模态组合
│   │   ├── diffusion/            #   Diffusion 模型（WAN、Qwen-Image）
│   │   └── common/               #   公共 Layer 与工具
│   ├── embodied/                 # LoongForge-Embodied：独立的 torch-native（DDP/FSDP）具身
│   │                             #   （VLA + 世界-动作）训练子系统，详见 loongforge/embodied/README_zh.md
│   ├── data/                     # 数据流水线（多模态、视频、DP 负载均衡）
│   ├── tokenizer/                # Tokenizer
│   └── utils/                    # 配置映射、常量等
├── third_party/Loong-Megatron/   # Patched Megatron-LM（git submodule）
├── configs/                      # Hydra YAML 配置（模型、数据）
├── examples/                     # GPU 启动脚本
├── examples_xpu/                 # 昆仑芯 XPU 启动脚本
├── tools/                        # Checkpoint 转换、数据预处理
├── ops/                          # 自定义融合算子（含开源的 TileLang 版本）
├── patches/                      # TransformerEngine 补丁
├── docker/                       # Dockerfile（GPU & XPU）
├── tests/                        # 端到端测试（YAML 驱动）
└── docs/                         # 文档
```

</details>

## 🤝 参与贡献

我们非常欢迎社区贡献 —— 无论是 Bug 报告、功能提案还是 PR。在提交前请阅读 [贡献指南](https://github.com/baidu-baige/LoongForge/blob/master/CONTRIBUTING.md)。

## 📄 开源协议

LoongForge 基于 [Apache License 2.0](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE) 发布。部分源文件改编自第三方开源项目，请以各文件头部标注的版权与署名信息为准。

## 📝 引用

如果您觉得 LoongForge 对您的工作有帮助，请引用本项目：

```bibtex
@software{LoongForge2026,
  title  = {LoongForge: A unified, high-performance framework for training LLMs, VLMs, diffusion, and embodied models},
  author = {{The LoongForge Authors}},
  year   = {2026},
  url    = {https://github.com/baidu-baige/LoongForge}
}
```

如果您在 LoongForge 中使用 TAOT 进行 MoE 训练，可以引用我们的论文：

```bibtex
@article{zhang2026taot,
  title   = {{TAOT}: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in {MoE} Training},
  author  = {Zhang, Lingyun and Zhang, Henghua and Gu, Shilei and Mo, Kai and Han, Shuai and Li, Shiyong and Wang, Yanpeng and Shen, Dou},
  journal = {arXiv preprint arXiv:2608.03676},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.03676}
}
```

## 🙏 致谢

LoongForge 的构建离不开 NVIDIA 的 Megatron-LM，并从 HuggingFace Transformers、LLaMA-Factory、Megatron-Bridge、LeRobot 等优秀开源项目，以及所支持模型的官方实现（如 OpenPI、NVIDIA Isaac GR00T）中汲取了灵感。衷心感谢这些社区所做的杰出贡献。

## 💬 联系我们
<a id="contact"></a>

欢迎通过 GitHub Issue 提交问题、反馈或功能建议，也可以加入我们的开发者社区：

- **微信群** — [扫码加入](https://github.com/baidu-baige/LoongForge/issues/80#issue-4594463290)
- **Slack** — [点击加入](https://join.slack.com/t/baiduloongforge/shared_invite/zt-3ys3kaq2p-cmdw0nDoaHGOcKibgys5Yw)
