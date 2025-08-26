# RLDecisionAgent: 使用强化学习训练决策大模型

本仓库是一个专注于使用强化学习（Reinforcement Learning）来训练大型语言模型（LLM）作为决策智能体（Decision-Making Agent）的项目集合。它包含了多个独立的框架和工具，旨在为大型模型的Agent能力提供训练、评估和优化的全套解决方案。

## 核心项目

本仓库包含以下几个核心项目的踩坑，每个项目都有其独特的侧重点和功能：

### 1. ART (Agent Reinforcement Trainer)

- **简介:** `ART` 是一个为真实世界任务训练多步Agent的强化学习框架。它利用 GRPO (Generative Reoptimization) 算法，让大型语言模型能从经验中学习，从而提升其可靠性。
- **主要特点:**
    - **MCP•RL:** 能够自动发现服务器工具、设计输入任务并训练模型以提升性能。
    - **无需标注数据:** 通过分析工具来学习任务。
    - **客户端/服务器架构:** 将训练服务器模块化，简化集成。
- **更多信息:** [ART/README.md](./ART/README.md)

### 2. AReaL (Ant Reasoning Reinforcement Learning)

- **简介:** `AReaL` 是一个开源的、完全异步的强化学习训练系统，专为大型推理模型设计。它致力于通过开源训练细节、数据和基础设施来帮助社区构建自己的AI Agent。
- **主要特点:**
    - **AReaL-lite:** 一个轻量级、算法优先的代码库，为AI研究人员提供更好的开发体验。
    - **完全异步:** 通过算法与系统的协同设计，实现极快的训练速度。
    - **可扩展性:** 能从单个节点无缝扩展到上千个GPU。
- **更多信息:** [AReaL/README.md](./AReaL/README.md)

### 3. Agent Lightning

- **简介:** `Agent Lightning` 是一个旨在点亮AI Agent的通用训练器。它最大的特点是几乎无需更改任何代码，即可将现有的Agent（无论基于何种框架）转化为一个可优化的模型。
- **主要特点:**
    - **零代码修改:** 只需在代码中添加两行，即可与任何Agent框架（如LangChain, AutoGen, CrewAI等）集成。
    - **选择性优化:** 在多Agent系统中，可以选择性地优化一个或多个Agent。
    - **支持多种算法:** 不仅限于强化学习，还支持自动提示优化（Automatic Prompt Optimization）等。
- **更多信息:** [agent-lightning/README.md](./agent-lightning/README.md)

### 4. verl
VERL (Volcano Engine Reinforcement Learning for LLMs)
* **简介:** `VERL` 是由字节跳动 Seed 团队发起、社区维护的面向大语言模型后训练的强化学习框架（HybridFlow 的开源实现）。它强调“灵活、高效、可用于生产”，可在少量代码中拼装出各类 RL 数据流。 ([GitHub][1], [Verl][2])
* **主要特点:**

  * **HybridFlow 编程模型：** 通过混合控制器（single/multi-controller）抽象，快速构建如 PPO、GRPO 等复杂后训练数据流。 ([GitHub][1], [Verl][2])
  * **生态与后端：** 训练后端支持 **FSDP / FSDP2 / Megatron-LM**；rollout 引擎支持 **vLLM、SGLang、HF Transformers**；可无缝对接 Hugging Face/ModelScope 上的主流开源模型（Qwen、Llama、Gemma、DeepSeek 等）。 ([GitHub][1])
  * **性能优化：** 集成 **3D-HybridEngine** 以在训练与生成阶段间高效重分片，降低内存冗余与通信开销；具备 SOTA 级吞吐。 ([GitHub][1], [Verl][2])
  * **可扩展并行：** 提供灵活设备映射与多机多卡扩展，内置多节点训练与 Ray 调试、性能调优指南。 ([Verl][2])
  * **算法与 Recipes：** 内置 **PPO、GRPO、GSPO、ReMax、REINFORCE++、RLOO、PRIME、DAPO、DrGRPO、KL\_Cov / Clip\_Cov、OPO、GPG** 等，并配套 **SPIN、SPPO** 等实用 recipe。 ([GitHub][1], [Verl][2])
  * **Agentic RL 支持：** 支持**服务器端异步 rollout**、**多轮对话与工具调用**，并提供 Agent Loop / 与 Agent 框架集成示例（如基于 LangGraph）。 ([GitHub][3], [Verl][2])
  * **硬件/平台：** 支持 **AMD ROCm** 与 **华为昇腾 Ascend**；Megatron 后端已用于大规模 **MoE**（如 DeepSeek-671B、Qwen3-236B）训练/推理场景。 ([Verl][2], [GitHub][1])
  * **工程能力：** 提供断点续训/容错检查点、RoPE scaling 覆写、LoRA、Profiler 与 Nsight 指南及 FAQ。 ([Verl][2])

### 5. AgentEvolutionRL

- **简介:** 这是一个总括性的项目，代表了本仓库的最终目标：通过强化学习训练和进化具备复杂决策能力的Agent大模型。
- **更多信息:** [AgentEvolutionRL/README.md](./AgentEvolutionRL/README.md)

## 环境设置与快速开始

为了方便开发和训练，提供了基于Docker的标准化环境。

### Docker镜像

```bash
docker pull vemlp-boe-cn-beijing.cr.volces.com/preset-images/verl:v0.4.1
```

### 创建并启动容器

使用以下命令创建一个共享主机网络的Docker容器。这使得容器可以直接访问主机的端口，无需额外映射。

```bash
# 创建容器，并挂载当前目录和时区
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name verl vemlp-boe-cn-beijing.cr.volces.com/preset-images/verl:v0.4.1 sleep infinity

# 启动容器
docker start verl

# 进入容器
docker exec -it verl bash
```

### 验证GPU

进入容器后，可以通过以下Python命令验证GPU是否可用：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
# 预期输出:
# True
# 1 (或你的GPU数量)
```

## 社区交流

欢迎和我微信进行交流！

![weichat.png](docs%2Fweichat.png)
