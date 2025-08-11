https://github.com/OpenPipe/ART/tree/main

OpenPipe 的 ART（Agent Reinforcement Trainer）是一个开源的强化学习框架，旨在通过经验学习提升大型语言模型（LLM）在多步骤任务中的可靠性。ART 采用 GRPO（Generalized Reinforcement Policy Optimization）方法，支持 Qwen、Llama、Kimi 等模型，适用于邮件检索、2048 游戏、MCP 协议等多种实际任务。([GitHub][1], [GitHub][2])

---

## 🔧 主要功能

* **MCP•RL**：通过强化学习自动训练模型，掌握任意 MCP（Model Context Protocol）服务器的工具和任务。
* **RULER**：一种自动化奖励生成系统，简化强化学习中的奖励设计。
* **AutoRL**：无需标注数据，通过自动输入生成和 RULER 评估，训练定制化 AI 模型。
* **ART·E**：训练 Qwen 2.5 14B 模型，在邮件检索任务中超越 OpenAI 的 o3。
* **多任务支持**：支持 2048、Tic Tac Toe、Codenames 等多种任务的训练。([GitHub][1])

---

## 🧪 示例与文档

ART 提供了多个示例和文档，帮助用户快速上手：

* [2048 游戏训练示例](https://github.com/OpenPipe/ART/blob/main/examples/2048/benchmark_2048.ipynb)
* [RULER 奖励系统文档](https://github.com/OpenPipe/ART/blob/main/docs/ruler.md)
* [ART·E 邮件检索案例](https://github.com/OpenPipe/ART/blob/main/docs/arte.md)

---

## ⚙️ 安装与配置

ART 可以通过以下命令安装：

```bash
pip install openpipe-art
```



如果需要使用 SkyPilot 后端，可安装带有 SkyPilot 支持的版本：([GitHub][3])

```bash
pip install openpipe-art[skypilot]
```



ART 支持与 W\&B、Langfuse 等平台集成，提供灵活的可观察性和调试功能。([GitHub][1])

---

## 📁 仓库结构

ART 仓库的主要目录包括：

* `src/`：核心代码实现。
* `examples/`：示例任务和 notebook。
* `docs/`：文档和教程。
* `.github/`：GitHub Actions 工作流配置。([GitHub][4], [GitHub][1])


# SkyPilot 是什么？

SkyPilot 是一个开源的多云资源管理和作业调度框架，目标是让分布式计算任务（比如强化学习训练、大模型训练）可以无缝地跨越多个云服务商和本地资源，方便地申请和调度计算资源。

它的特点包括：

* **多云兼容**：支持 AWS、Azure、GCP 等主流云平台资源。
* **自动调度**：根据用户需求，自动分配和调度计算实例。
* **弹性伸缩**：可根据任务负载动态增减资源。
* **统一接口**：用户只需用一套配置，即可在不同云上运行分布式任务。

---

### 在 ART 中的作用

ART 框架可以用 SkyPilot 作为其后端运行环境，这意味着：

* **训练任务的资源调度由 SkyPilot 管理**，无需用户手动去开云服务器。
* 可以**方便地在多云环境中弹性扩展训练规模**，提升训练效率。
* 支持自动申请和释放云端实例，减少资源浪费。
* 使得强化学习训练更加自动化和便捷，尤其是大规模分布式训练。

---

### 简单总结

> SkyPilot 后端就是 ART 里用来申请和管理训练所需计算资源的“云资源管理器”，帮助你自动找、申请、调度云上或者本地的计算机器，让强化学习训练更省心、省力。

---

如果你要用 ART 在云上跑大规模训练，装上带 SkyPilot 支持的版本（`pip install openpipe-art[skypilot]`）就可以直接用 SkyPilot 来管理资源了。你还想了解 SkyPilot 怎么具体用或者怎么配置吗？
