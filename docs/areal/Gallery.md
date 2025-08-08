# RLVR解释
在 AReaL 框架中，术语 **RLVR** 指的是 **Reinforcement Learning with Verifiable Rewards**，即“具有可验证奖励的强化学习”。这是一个在推理任务中用于训练大型语言模型 (LLMs) 的常见训练范式，它的关键特征包括：

* 使用一个可以明确判断答案是否正确的“验证器”（如数学题有正确答案、代码生成可以通过单元测试）来为模型生成的结果提供奖励（正确为 1，错误为 0），从而替代传统 RL 中依赖于经过训练的奖励模型（reward model）([rlhfbook.com][1], [arXiv][2])。

* 它与 RLHF（通过偏好训练奖励模型）类似，但更注重提供直接的、可验证的信号，使得在具备明确正确输出的任务中性能提升显著([rlhfbook.com][1], [Medium][3])。

在 AReaL 项目中，RLVR 通常与其异步训练系统结合使用，支持用户在 AReaL 架构内实现自定义的 agent 行为或 RLVR rollout 流程。例如，在 AReaL‑lite 版本的代码中，你可以在一个文件里找到有关 “agentic & RLVR rollout” 的定制脚本或配置方式([GitHub][4])。

总结一下：

**RLVR = Reinforcement Learning with Verifiable Rewards（具有可验证奖励的强化学习）**

是在有清晰验证机制（如正确答案、测试脚本等）任务上常用的 RL 训练方式，AReaL 将其与异步训练机制整合，用于提升推理性能及训练效率。


# SPMD(单程序，多数据)
Single Program, Multiple Data
“SPMD（Single Program, Multiple Data）” 是在大规模深度学习训练中广泛推荐的一种并行计算模式。它的核心思想是：
所有设备（例如多个 GPU）运行相同的程序，但每个设备处理不同的数据或模型的不同部分。
“**SPMD（Single Program, Multiple Data）**” 是在大规模深度学习训练中广泛推荐的一种并行计算模式。它的核心思想是：

> 所有设备（例如多个 GPU）**运行相同的程序**，但每个设备处理不同的数据或模型的不同部分。

---

## ✅ SPMD 的主要优点

### 1. **易于编程与维护**

* 所有设备运行的是 **同一份代码（单程序）**，没有主-从角色划分（不像某些 MPI 模式）。
* 降低了逻辑复杂度，不需要为每张卡写不同逻辑。

### 2. **良好的可扩展性**

* SPMD 模式天然支持 **数据并行（DP）**、**张量并行（TP）**、**模型并行（MP）** 等多种并行策略。
* 在多个 GPU 上能够很好地扩展，例如 PyTorch 的 `torch.distributed` 和 `torchrun` 就是 SPMD 架构。

### 3. **兼容现代训练框架**

* 多数训练工具链（如 DeepSpeed、FSDP、Megatron-LM、Colossal-AI、SGLang 等）都采用或推荐使用 SPMD。
* 简化了分布式通信、调度、同步等问题。

### 4. **高性能潜力**

* SPMD 在搭配通信优化（如 NCCL、Gloo）时，可以实现高效的数据同步。
* 更适合在 GPU 集群中进行优化调度（如流水线并行 + 张量并行）。

### 5. **更容易调试与迁移**

* 因为程序一致，调试时只需要关注一份代码的行为。
* 在单卡、多卡、多机迁移时不需要重新写逻辑，改下启动参数即可。

---

## 📌 举例：在 PyTorch 中使用 SPMD

```bash
torchrun --nproc_per_node=8 train.py
```

所有 8 个 GPU 会运行 `train.py`，但每张卡通过 `rank` 来知道自己是哪个分片，该加载哪些数据、参数。

---

## 🚫 对比：非 SPMD 的缺点

* 有些旧系统采用“master-worker”模式，每个设备跑不同代码，增加调试/管理难度。
* 不易扩展到多节点训练。

---

## ✅ 总结一句话：

> **SPMD = 简洁、可扩展、高性能、现代分布式训练的主流模式。**
如果你在用 PyTorch、JAX、SGLang、FSDP 等系统，大概率都推荐使用 SPMD 模式。如果你要部署/训练大模型，这种模式几乎是必须的。



# 共卡
训推共卡（Training–Inference Co‑card）：同一计算卡同时承担训练与推理任务。这种方式可以提升资源利用率，降低训练/推理切换延迟。例如华为的 RLFusion 技术就支持该模式，能让同一张卡“一箭双雕”，即训练与推理并行进行，显著提升效率

# `sglang.d8p1t1+d8p1t1` 是一种 **资源分配配置字符串**，通常用于说明模型部署时在硬件上的分布方式，尤其是**GPU 分配策略**。这类格式一般出现在基于 [SGLang](https://github.com/InternLM/sglang) 或类似系统中，用于定义 Actor、Decoder、Tokenizer 的资源使用情况。

---

### 拆解 `sglang.d8p1t1+d8p1t1` 含义：

这个字符串由两个部分组成，用 `+` 连接：

#### ✅ `sglang.d8p1t1`：

* `sglang`：使用的后端/系统类型（比如 HuggingFace、vLLM、SGLang）。
* `d8p1t1` 是资源分配的关键：

  * `d8` → decoder 使用 **8 个 GPU**。
  * `p1` → pipeline 并行度为 1。
  * `t1` → tokenizer 线程为 1。

#### ✅ `+d8p1t1`：

* 这是对第二个模块的相同资源配置（比如用于训练或推理时的另一个服务组件）。
* 没有再写 `sglang.` 前缀，通常意味着和前一个相同。

---

### 举个例子：

假设你部署一个模型服务系统，有两部分：

* 一个 **Actor 模型服务**，处理推理或训练。
* 一个 **Reward 模型/价值模型服务**（或者多个 Actor 并行服务）。

那么 `allocation_mode: sglang.d8p1t1 + d8p1t1` 表示：

* 每个模型实例占用 8 个 GPU（总共至少需要 16 个 GPU）。
* 每个实例不使用 pipeline 并行，只是使用单个 pipeline 和 tokenizer。

---
### 结论总结：

> `sglang.d8p1t1+d8p1t1` 表示：使用 SGLang 系统，部署两个服务实例，每个实例使用 8 张 GPU，pipeline 和 tokenizer 都是 1，适用于中大型模型并行部署。


# GPU的资源分配

### 🔧 常见分配模式对照表

| 模型尺寸 (Model Size) | GPU 数 (GPUs) | 分配模式 (Allocation Mode)     | 描述 (Description)                              | 结构说明                                                                                                     |
| ----------------- | ------------ | -------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **1.5B**          | 8            | `sglang.d4p1m1 + d2p2m1`   | 单节点，最小并行度<br>Single node, minimal parallelism | - Decoder: 4 GPU<br>- Pipeline: 1<br>- Model parallel: 1<br>+ 第二实例: 2 GPU，2 pipeline stage，1 model shard |
| **1.5B**          | 32           | `sglang.d16p1m1 + d8p2m1`  | 多节点数据并行<br>Multi-node data parallel           | 主体模型用 16 GPU，另一个服务用 8 GPU<br>pipeline 和模型并行都比较轻                                                          |
| **7B**            | 32           | `sglang.d16p1m1 + d8p2m1`  | 平衡数据/流水线并行<br>Balanced DP/PP                  | 和上面一样，适合中等规模模型训练或部署                                                                                      |
| **7B**            | 128          | `sglang.d64p1m1 + d32p2m1` | 大规模数据并行<br>Large-scale data parallel          | 多节点数据并行，主模型用 64 GPU，次模型用 32 GPU，适合预训练                                                                    |
| **32B**           | 128          | `sglang.d8m8p1 + d4p4m4`   | 重模型并行+流水线<br>Heavy MP + PP                    | 主模型使用 8-way 模型并行 + 8-stage pipeline<br>适合超大模型部署                                                          |
| **32B (SFT)**     | 128          | `d32p1m4`                  | 仅训练，无 SGLang<br>Training-only, no SGLang      | 没有 `sglang.` 前缀，表示不用 SGLang 服务，仅训练使用<br>32 GPU，单 pipeline，4-way 模型并行                                     |

---

### 📘 缩写说明

* `dX`: Decoder 使用 X 张 GPU（通常也代表数据并行度）
* `pX`: Pipeline 并行度为 X
* `mX`: 模型并行度（Model parallel）为 X
* `+`: 表示复合部署（如多模型实例）

---

### ✅ 建议使用场景

* **小模型（1.5B）**：单节点多卡即可，低复杂度。
* **中模型（7B）**：需要多节点，开始使用更多 pipeline 并行。
* **大模型（32B）**：必须启用 pipeline + model parallel，甚至跨多个节点部署。
* **训练专用（如 SFT）**：可以不使用 `sglang`，只关注数据/模型/流水线并行即可。