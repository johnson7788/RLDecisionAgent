# Parquet
Parquet 是一种列式存储格式（Columnar Storage Format），专为高效存储和处理大规模数据而设计，特别适合大数据处理和分析场景。它是 Apache 开发的开源项目，常用于 Hadoop、Spark、Hive、Presto、Pandas 等数据处理系统中。
```python
import pandas as pd

# 保存 DataFrame 为 Parquet
df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
df.to_parquet('train.parquet', engine='pyarrow')

# 读取 Parquet 文件
df2 = pd.read_parquet('train.parquet', engine='pyarrow')
print(df2)
```

# 训练命令解析
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo ...
PYTHONUNBUFFERED=1：输出不缓冲，确保训练日志实时显示。

python3 -m verl.trainer.main_ppo：运行 PPO 强化学习训练模块。

数据配置
data.train_files=/home/.../train.parquet
data.val_files=/home/.../test.parquet
data.train_batch_size=16
data.max_prompt_length=512
data.max_response_length=256
指定训练数据与验证数据的 Parquet 文件路径。

设置训练 batch 大小为 16。

限制模型输入（prompt）最大长度为 512，输出（response）最大长度为 256。

Actor 模型配置
actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

actor_rollout_ref.model.path：使用 Qwen2.5-0.5B-Instruct 作为 Actor 模型。

lr=1e-6：Actor 的学习率设为 1e-6。

mini_batch 是 PPO 每轮迭代使用的小批量大小；micro_batch 是在 GPU 上实际执行的 batch 大小（用于显存控制）。


Rollout 配置（生成动作/采样）
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
控制 rollout 阶段每 GPU 的 batch 大小（用于计算 log probability）。

指定并行度为 1（不开 tensor/model 并行）。

使用 GPU 显存的最大比例为 0.4（防止 OOM）

Reference 模型配置
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
reference model 负责对输出进行打分（log_prob），用于计算奖励和 KL loss。

Critic 模型配置
critic.model.path=Qwen/Qwen2.5-0.5B-Instruct
critic.optim.lr=1e-5
critic.ppo_micro_batch_size_per_gpu=4
使用同样的模型作为 critic。

critic 学习率为 1e-5。

critic 在每个 GPU 上使用的 micro batch size。

algorithm.kl_ctrl.kl_coef=0.001
PPO 中用于控制 Actor 和 Reference 之间的差异的 KL 惩罚系数。越大，Actor 越不敢偏离 Reference。


Trainer配置
bash
复制
编辑
trainer.logger=console
trainer.val_before_train=False
trainer.n_gpus_per_node=1
trainer.nnodes=1
trainer.save_freq=10
trainer.test_freq=10
trainer.total_epochs=3
使用 console 日志器（而非 wandb 等）。

val_before_train=False：跳过训练前的验证。

使用 1 张 GPU，单节点训练。

每训练 10 个 epoch 进行一次保存和测试。

总共训练 3 个 epoch。


# Actor（演员）
核心模型，在训练中不断调整权重以优化回答质量。

在每个训练步骤中，接受一个 prompt（输入），然后生成 response（输出）。

模型的行为会根据 reward（由 critic 给出）来更新，使其更符合人类偏好或目标任务。

Actor 会尝试生成策略 π_θ，并通过 PPO 算法进行稳定更新，使其比 Reference 更好（更高 reward）。

# Critic（评论家）
负责给 Actor 的输出打分（即 reward）
对每个 prompt 和 Actor 输出的 response，预测一个 reward 分数（用作 PPO 中的价值函数 V(s)）。

Critic 本质上是一个 回归模型，输出 reward 期望。

它是监督 Actor 学会什么样的回答是“好的”的依据。

# Reference（参考模型）
 Reference 模型：用来限制 Actor 偏离太远
冻结不更新，作为 Actor 的旧版本。

用来计算 KL 散度（Actor 的输出与 Reference 的差距），防止模型训练过度偏离初始能力（即保持“原样性”）。

越偏离，惩罚越大（通过 KL loss），控制模型学习的“激进程度”。

Prompt ─────────► Actor ───────────────► Response
                    │                        │
                    ▼                        ▼
             Reference Model          Critic（打分）
                    │                        │
                    ▼                        ▼
             KL divergence             Reward Value
                    │                        │
                    └──────┬────────────┬────┘
                           │            │
                    PPO 更新策略       PPO 更新Critic


Maxwell-Jia/AIME_2024


# verl配置文件路径
[evaluation.yaml](..%2Fverl%2Fverl%2Ftrainer%2Fconfig%2Fevaluation.yaml)
[generation.yaml](..%2Fverl%2Fverl%2Ftrainer%2Fconfig%2Fgeneration.yaml)
[ppo_megatron_trainer.yaml](..%2Fverl%2Fverl%2Ftrainer%2Fconfig%2Fppo_megatron_trainer.yaml)
[ppo_trainer.yaml](..%2Fverl%2Fverl%2Ftrainer%2Fconfig%2Fppo_trainer.yaml)
[sft_trainer.yaml](..%2Fverl%2Fverl%2Ftrainer%2Fconfig%2Fsft_trainer.yaml)

# RL的数据集中的interaction_kwargs字段的意思
https://verl.readthedocs.io/en/latest/sglang_multiturn/interaction_system.html?utm_source=chatgpt.com
与特定样本对应的交互逻辑参数， Rollout 阶段（sglang_rollout.py） ，在实际 rollout 过程中，当请求状态为 INTERACTING 时，系统会读取 _req.interaction_kwargs 中的 "name" 字段来选择交互 agent：
然后调用对应的交互类实例引导多轮对话、提供反馈、计算奖励等，verl 交互系统在强化学习训练期间支持动态、多轮对话反馈。该系统允许模型参与迭代问题解决场景，交互代理可以根据模型的响应提供纠正反馈、指导或评估。
参考： verl/interactions/gsm8k_interaction.py
Verl 框架中用于 GSM8K 任务的交互代理类 Gsm8kInteraction，它继承自 BaseInteraction，用于指导训练模型在 RLHF 或 DPO 过程中通过多轮交互方式提升数学题的解答能力。
💡 interaction_kwargs 在哪体现？
交互流程是围绕样本携带的 interaction_kwargs 来配置的，例如：
{
  "name": "gsm8k",
  "query": "Samantha has 12 apples, eats 3...",
  "ground_truth": "The correct answer is 9."
}
Verl 在 rollout 阶段：
调用 interaction = interaction_map["gsm8k"]
用 start_interaction(ground_truth="The correct answer is 9.") 启动状态
模型输出后，generate_response() 判断答题对错
给出奖励和环境反馈（用于下一步训练或 sampling）