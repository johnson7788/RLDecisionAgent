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

# 共卡
训推共卡（Training–Inference Co‑card）：同一计算卡同时承担训练与推理任务。这种方式可以提升资源利用率，降低训练/推理切换延迟。例如华为的 RLFusion 技术就支持该模式，能让同一张卡“一箭双雕”，即训练与推理并行进行，显著提升效率