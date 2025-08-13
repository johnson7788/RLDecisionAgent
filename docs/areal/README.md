# 仓库
https://github.com/inclusionAI/AReaL

# 智星云 + Areal
docker pull ghcr.io/inclusionai/areal-runtime:v0.3.0.post2

# 训练基础条件
至少2张独立显卡，一张用于推理，一张用于训练

## 2. AReaL-lite 的特点
算法优先（algorithm-first），而不是旧版的“系统优先”。

轻量化：代码量比旧版少 80%，但保留 90% 的性能和功能。

易定制：在一个文件里就能定制算法、Agentic 工作流、RLVR（RL with Verifiable Rewards）。

可扩展：无需懂系统细节，就能从单机扩展到分布式集群。

提供了代码走读示例（GRPO 在 GSM8K 数据集上的训练）。

## 3. 核心功能亮点
全异步 RL：不必等待所有节点同步，可以让采样和训练同时进行，大幅提速。

支持多轮 Agentic RL：让模型学会多轮推理和工具调用。

高性能与可扩展性：单命令启动，从单节点到 1000+ GPU。

最前沿性能：数学推理、编程任务效果优秀。


## 单机运行示例：
python3 -m areal.launcher.local examples/lite/gsm8k_grpo.py --config examples/lite/configs/gsm8k_grpo.yaml
