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

## evaluation.yaml
data:
  path: /tmp/math_Qwen2-7B-Instruct.parquet
  prompt_key: prompt
  response_key: responses
  data_source_key: data_source
  reward_model_key: reward_model
path：指定评估数据文件路径，应为 .parquet 文件。

prompt_key：用于从数据中提取输入提示的字段名。

response_key：表示模型生成响应（生成的答案）字段，通常是字符串列表。

data_source_key：用于区分不同数据来源，在评估时可分别计算各来源的指标。

reward_model_key：代表“参考答案”或评分模型输出的字段名，用于与生成输出做对比。


custom_reward_function:
  path: null
  name: compute_score
path：指定包含自定义奖励函数的 Python 文件路径。如果为 null，将使用内置的预设函数。
name：函数名，默认是 compute_score。如果你只写一个 compute_score 函数，可以简单使用默认即可。

ray_init（与 Ray 初始化相关）
ray_init:
  num_cpus: null
  timeline_json_file: null
num_cpus：用于控制 Ray 集群使用的 CPU 核心数。若为 null（或 None），默认使用系统所有 CPU，但在一些集群环境（如 SLURM）可能会导致卡住。建议明确设置一个允许使用的数字。
timeline_json_file：可选路径，用于输出 Ray Timeline 的 JSON 文件，便于调试性能问题。若不需要则设为 null。、

# generation.yaml
下面是对你提供的 `generation.yaml` 配置逐项解释，结合 Verl 官方文档中的说明（截至 2025 年更新）([Verl][1])。


## 🧠 Trainer 设置

```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 8
  device: cuda
```

* **`nnodes`** 与 **`n_gpus_per_node`**：配置用于生成的节点数和每节点 GPU 数量，支持多节点训练／generation。
* **`device`**：使用 GPU (`cuda`) 而非 CPU。

---

## 📂 Data 部分

```yaml
data:
  path: ~/data/rlhf/math/test.parquet
  prompt_key: prompt
  n_samples: 5
  output_path: /opt/tiger/math_Qwen2-7B-Instruct.parquet
  batch_size: 128
```

* **`path`**：输入数据源，通常是 `.parquet` 格式。
* **`prompt_key`**：数据中的提示字段名（prompt）。
* **`n_samples`**：每个 prompt 生成多少个答案样本（这里为 5）。
* **`output_path`**：生成的样本将保存到该路径。
* **`batch_size`**：一次处理多少 prompt，以提高并行吞吐。

---

## 🧱 Model 配置

```yaml
model:
  path: ~/models/Qwen2-7B-Instruct
  external_lib: null
```

* **`path`**：指定用于推理或生成的模型路径。
* **`external_lib`**：如为 null 使用默认库，也可以指定自定义库（例如有特殊 tokenizer 或后处理）。

---

## 🔄 Rollout 模块

```yaml
rollout:
  name: vllm
  mode: sync
  temperature: 1.0
  top_k: 50
  top_p: 0.7
  prompt_length: 1536
  response_length: 512
  dtype: bfloat16
  gpu_memory_utilization: 0.5
  ignore_eos: False
  enforce_eager: True
  free_cache_engine: True
  load_format: dummy_dtensor
  tensor_model_parallel_size: 1
  max_num_batched_tokens: 8192
  max_num_seqs: 1024
  log_prob_micro_batch_size_per_gpu: 8
  do_sample: True
  disable_log_stats: True
  enable_chunked_prefill: True
  n: 1
  calculate_log_probs: False
```

### 🎯 基础采样参数

* **`name: vllm`**：选择 vLLM 作为 rollout engine。
* **`mode: sync`**：同步模式（async 表示使用 AsyncLLM）。
* **`temperature`、`top_k`、`top_p`**：控制采样策略的随机性与多样性([Verl][1], [vLLM Forums][2])。

### ⚙️ vLLM 特定设置

* **`dtype: bfloat16`**：指定生成使用的浮点类型，与训练 actor 模型保持一致([Verl][1])。
* **`gpu_memory_utilization: 0.5`**：vLLM 占用 GPU 总内存的比例，通常设 0.5–0.7 以平衡吞吐与 OOM 风险([Verl][1])。
* **`ignore_eos`**：不在生成结束时因遇 EOS token 而停止。
* **`enforce_eager`**：关闭 CUDA 图（CUDAGraph），避免 vLLM 某些版本在缓存释放过程中崩溃([Verl][1])。
* **`free_cache_engine`**：生成后释放 KV cache，配合 `enforce_eager=True` 以降低内存。
* **`load_format`**：`dummy_dtensor` 用于 FSDP 后端的虚拟初始化方式，延迟实现权重同步([Verl][1])。
* **`tensor_model_parallel_size: 1`**：TP size，只使用 1 份 vLLM 副本。

### 批量调优参数

* **`max_num_batched_tokens: 8192`** 和 **`max_num_seqs: 1024`**：控制每次生成时处理的 token 和序列数。增大可提高吞吐性能([Verl][1], [Verl][3])。
* **`log_prob_micro_batch_size_per_gpu: 8`**：每个 GPU 用于 log‑prob 计算的小批量大小（替代过时 `log_prob_micro_batch_size`）([Verl][1])。

### HF Rollout 模式参数（兼容性）

* **`do_sample`**：是否采样（不 greedy）。
* **`disable_log_stats`**、**`enable_chunked_prefill`**：可用于统计日志控制与内存分块预填充优化。
* **`n`**：每个 prompt 输出条数，通常为 1。

### 调试选项

* **`calculate_log_probs: False`**：是否在生成过程中记录 log‑prob，方便调试但会影响性能。

---

## 🎬 Actor 模型细节（Actor Rollout）

```yaml
actor:
  strategy: fsdp
  ulysses_sequence_parallel_size: 1
  entropy_from_logits_with_chunking: False
  entropy_checkpointing: False
  fsdp_config:
    fsdp_size: -1
    forward_prefetch: False
```

* **`strategy: fsdp`**：使用 PyTorch FSDP 后端训练 actor。
* **并行与熵计算相关设置**（如 Ulysses parallel, checkpointing）是用于内存优化和吞吐提升的细节选项。

---

## ☁️ Ray 初始化（与 evaluation.yaml 一致）

```yaml
ray_init:
  num_cpus: null
  timeline_json_file: null
```

* **`num_cpus: null`**：默认使用系统全部 CPU，建议在 SLURM 等集群环境中设置为特定值以避免 hang。
* **`timeline_json_file`**：可写入 Ray 性能 timeline JSON，以用于调试。

---

## ✅ 总结一览

| 模块            | 配置项                             | 说明              |
| ------------- | ------------------------------- | --------------- |
| **trainer**   | nnodes / n\_gpus\_per\_node     | 多节点与 GPU 数设置    |
|               | device                          | 使用 GPU 或 CPU    |
| **data**      | n\_samples / batch\_size        | 多样本生成及并行吞吐      |
| **model**     | path / external\_lib            | 模型路径与自定义库       |
| **rollout**   | name, mode                      | rollout 引擎与调用模式 |
|               | temperature, top\_k, top\_p     | 生成策略控制          |
|               | dtype, gpu\_memory\_utilization | 内存类型及占用比例       |
|               | load\_format                    | 权重加载方式匹配训练后端    |
|               | batched\_tokens, seqs           | 批处理规模控制         |
| **actor**     | fsdp / Ulysses parallel 等       | actor 训练行为与优化开关 |
| **ray\_init** | num\_cpus, timeline\_json\_file | Ray 资源控制与调试辅助   |





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


# verl中的multiturn多轮对话是如何计算奖励的？
多轮对话（multi‑turn dialogue）的奖励机制通常是基于 每个对话回合（turn-level） 或 整个对话轨迹（trajectory-level） 
| 方法类别                      | 奖励时机           | 奖励来源                                   | 适用场景                          |
| ------------------------- | -------------- | -------------------------------------- | ----------------------------- |
| **回合级（turn-level）**       | 每个回合后          | tool 打分，如 correctness                  | GSM8K、QA 每步打分型对话              |
| **轨迹级（trajectory-level）** | 对话结束后          | 最终是否正确、judge 模型输出                      | MGPO、ARTIST、agent agents 类 RL |
| **混合模型**                  | 兼具回合内评分与最终轨迹奖励 | proxy 或 correctness + information gain | 信息稀疏、多回合推理任务                  |

# RL训练集中的agent_name字段
https://verl.readthedocs.io/en/latest/start/agentic_rl.html#overview
Tool Agent Loop 要求向数据集添加“agent_name”字段。在转出期间，它将根据此字段选择使用 tool_agent_loop 或 single_turn_agent（默认）。
RL 数据集中的 agent_name 字段主要用于指定训练时所使用的代理循环策略（agent loop）,在多轮交互（multi-turn conversation）和工具调用（tool calls）场景中，Tool Agent Loop 要求数据集里必须包含 "agent_name" 字段。系统会依据该字段值 决定使用 tool_agent_loop 还是 single_turn_agent（默认） 进行后续 rollout 处理
| 文件 / 用例                                                | agent\_name 值         | 作用                           |
| ------------------------------------------------------ | --------------------- | ---------------------------- |
| GSM8K 工具调用训练脚本 (`gsm8k_tool_agent_loop.py`)            | `"tool_agent"`        | 使用工具代理循环，开启工具调用支持与计算奖励融合     |
| Multi‑turn React Agent 测试 (`test_react_agent_loop.py`) | `"react_agent"`       | 测试带有多轮 React 行为的 agent loop  |
| 单轮测试 (`test_basic_agent_loop.py`)                      | `"single_turn_agent"` | 使用单回合代理逻辑，无工具调用参与            |
| 数学表达式数据生成 (`create_dataset.py`)                        | `"math_expression"`   | 自定义 agent loop 类型，用于特定数学表达任务 |

tool_agent：用于 GSM8K 数学题 + 工具调用训练流程，会使用 ToolAgentLoop 类来支持 step‑by‑step 推理、调用 calc_gsm8k_reward 等工具逻辑。

react_agent：在 React Agent（LangGraph 风格）中使用，用于 multi‑turn 工具调用 + 自反思推理类型流程。

single_turn_agent：默认单轮响应 agent loop，无工具调用能力，适用于简单一次性回答流程。

math_expression：自定义 agent loop，例如用于生成数学表达式任务的流程逻辑。

cat ./verl/verl/experimental/agent_loop/__init__.py

./verl/verl/experimental/agent_loop/tool_agent_loop.py
verl 中的一个核心 agent loop runner 类型：ToolAgentLoop。它是一个基于工具调用的多轮对话 Agent，用于在强化学习训练或推理时，自动处理工具调用、生成响应、以及与环境交互。

每轮循环包括：

调用 LLM 生成响应（assistant 回合）。

解析响应中是否包含工具调用。

如果有工具调用，就执行工具并生成 tool response。

将 tool response 拼接进 prompt，作为下一轮输入（user 回合）。

直到达到这些终止条件之一：

达到最大 token 限制（response_length）

达到最大 assistant turn 或 user turn

没有工具调用了

工具调用异常（如报错


## ppo_trainer.yaml
这个 `ppo_trainer.yaml` 是 **VERL**（Versatile RLHF Library）框架中用于配置 **PPO（Proximal Policy Optimization）训练器** 的核心配置文件，配置内容较多，下面我会从整体结构和关键字段解释其作用与设计思路。

---

## 🧭 文件整体结构概览

```yaml
defaults:
  - actor@actor_rollout_ref.actor: dp_actor
  ...
  - _self_
```

VERL 使用 Hydra/OmegaConf 的配置继承系统，`defaults` 块用来定义配置组合方式，即：

* 每一项 `<子模块>@<路径>` 表示将一个 yaml 文件绑定到当前配置的子模块上；
* `_self_` 表示当前这个 `ppo_trainer.yaml` 可以覆盖前面的默认设置。

---

## 🧠 核心模块分解

---

### 1. `actor_rollout_ref`: Actor、Rollout、Reference模型配置

这个模块统一管理：actor 模型（训练用）、rollout 模型（生成用）、reference 模型（用来算KL）。

```yaml
actor_rollout_ref:
  hybrid_engine: true  # 使用混合引擎（actor、rollout、ref共存）
  model: {...}         # 模型加载方式、LoRA、是否开启 gradient checkpointing
  rollout: {...}       # rollout专用的优化配置
  profiler: {...}      # profiler设置
```

主要字段解析：

#### `model.path`

模型路径，可以是本地或远程 HuggingFace 模型。

#### `lora_rank`, `lora_alpha`, `target_modules`

LoRA 的配置，控制是否低秩微调，哪些模块应用 LoRA。

#### `enable_gradient_checkpointing`

节省内存，提高 batch size（但会减慢训练速度）。

#### `rollout.enable_chunked_prefill`

是否开启 chunked prefill（大模型时能显著提高吞吐量）。

---

### 2. `custom_reward_function`: 自定义奖励函数设置

```yaml
custom_reward_function:
  path: null
  name: compute_score
```

* `path`: 可填入一个 Python 文件路径，自定义奖励函数；
* `name`: 奖励函数名称，默认使用 `compute_score`。

---

### 3. `algorithm`: PPO算法的关键超参数配置

```yaml
algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: gae
  use_kl_in_reward: False
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
```

解释几个关键参数：

| 参数名                | 说明                                                    |
| ------------------ | ----------------------------------------------------- |
| `gamma`            | 折扣因子，越低越短视；1.0 代表不折扣未来奖励。                             |
| `lam`              | GAE (Generalized Advantage Estimator) 的权衡系数。          |
| `adv_estimator`    | 优势函数估计方式，比如 `"gae"`、`"reinforce_plus_plus"` 等。        |
| `use_kl_in_reward` | 是否将KL散度作为奖励惩罚项。                                       |
| `kl_ctrl`          | KL控制策略：可以是 `fixed` 或 `adaptive`；adaptive 可用于动态调节KL系数。 |

---

### 4. `trainer`: PPO主训练器配置

```yaml
trainer:
  total_epochs: 30
  n_gpus_per_node: 8
  logger: ['console', 'wandb']
  project_name: verl_examples
  experiment_name: gsm8k
  resume_mode: auto
  val_before_train: True
  test_freq: -1
```

重点字段说明：

| 字段                                        | 含义                 |
| ----------------------------------------- | ------------------ |
| `total_epochs`                            | 总共训练 epoch 数       |
| `save_freq`                               | 多久保存一次模型（按 step）   |
| `logger`                                  | 日志输出后端，如控制台或 wandb |
| `rollout_data_dir`, `validation_data_dir` | rollout/val的生成输出目录 |
| `resume_mode`                             | 自动恢复训练             |
| `val_only` / `val_before_train`           | 控制验证行为             |

关于 **Nsight GPU profiling**：

```yaml
profile_steps: null
controller_nsight_options:
  trace: "cuda,nvtx,cublas,ucx"
  cuda-memory-usage: "true"
```

只有在 `profile_steps` 指定时才会生效，用于 GPU 性能诊断分析。

---

### 5. `ray_init`: Ray分布式初始化配置

```yaml
ray_init:
  num_cpus: null
  timeline_json_file: null
```

通常用于分布式训练环境，若使用 SLURM 建议显式指定 `num_cpus`。

---

## 💡 实战建议

| 需求            | 推荐设置                                       |
| ------------- | ------------------------------------------ |
| **节省显存**      | 开启 `enable_gradient_checkpointing: true`   |
| **训练多轮对话任务**  | 设置 `use_kl_in_reward: true`，开启参考模型KL惩罚     |
| **调试**        | 只启用 console 日志 + 设置较小 batch                |
| **LoRA微调**    | 设置 `lora_rank > 0` 并配置合适的 `target_modules` |
| **Profiling** | 开启 `profile_steps` 并配置 `nsight_options`    |


# HybridFlow原理

## 🧠 整体理解：HybridFlow 在代码中的体现

HybridFlow 的目标是将 RLHF 训练流程（如 PPO）表示为灵活的数据流图，再将每个子模块（Actor、RewardModel、Critic 等）封装为并行计算节点，并通过混合控制器调度它们。Verl 的代码结构完美反映了这一设计。

---

## 🔧 核心模块讲解

### 1. **训练入口与调度控制器**

```
verl/trainer/main_ppo.py
verl/trainer/ppo/ray_trainer.py
```

* **main\_ppo.py**：训练入口，加载配置、初始化模型、训练器。
* **ray\_trainer.py**：

  * 封装 PPO 等算法的主训练循环；
  * 调用 rollout、reward、critic、update 等步骤；
  * 每一步都是一个「flow step」，被送入 HybridFlow 的调度图中。

⚙️ 调用方式大致是：

```python
rollouts = actor.generate()
rewards = reward_model.score(rollouts)
values = critic.evaluate(rollouts)
updated_actor = ppo.update(actor, rewards, values)
```

这些操作实际是 **由 HybridFlow DAG 结构组织起来的任务流**。

---

### 2. **Workers 模块：执行单元实现**

```
verl/workers/
```

这是执行物理计算任务的“工人层”模块：

* `actor/`：负责文本生成（Actor 模型的 rollout）。

  * `dp_actor.py`: 使用 FSDP 的数据并行策略。
  * `megatron_actor.py`: 使用 Megatron 的 nD 并行策略。

* `critic/`：用于值函数估计。

* `reward_model/`：打分模块。

* `rollout/`：接入不同推理后端（如 vllm, TGI）的接口。

⛓️ 它们通过 `protocol.py` 中的接口标准化为 HybridFlow 可调度任务，成为 “数据流节点”。

---

### 3. **Sharding Manager：高效切换与模型重分布**

```
verl/workers/sharding_manager/
```

用于支持 **Actor 模型在 rollout ↔ 训练阶段之间的高效切换**，是 HybridFlow 的效率核心：

* `fsdp_ulysses.py`: 在 FSDP 下进行权重重分布；
* `megatron_vllm.py`: Megatron + vllm 场景下的 shard 操作。

📌 例如，rollout 阶段我们需要 fp16、无梯度的推理模型；
训练阶段需要全精度梯度模型，重 shard 就发生在这两者之间。

---

### 4. **配置系统（生成 + 训练）**

```
verl/config/
  generation.yaml
  ppo_trainer.yaml
```

统一参数入口，描述：

* 使用哪个模型
* 使用哪个 worker backend
* rollout 的 batch size、设备映射等
* PPO 或 GRPO 的训练参数

所有这些会影响 HybridFlow DAG 的形状与调度策略。

---

### 5. **utils + reward\_score + dataset**

```
verl/utils/
verl/utils/reward_score/
verl/utils/dataset/
```

这些模块提供：

* Dataset 加载与预处理；
* 自定义奖励函数（如 math、gsm8k）；
* Sequence length balancing 等优化技巧。

这些会参与 `rollout → reward` 这一子流程的数据变换。

---

## ✅ HybridFlow 的关键创新在此体现：

| 模块        | 对应原理                        | 实现位置                                          |
| --------- | --------------------------- | --------------------------------------------- |
| 任务图抽象     | RLHF 训练流建模为数据流图             | `ray_trainer.py`, `protocol.py`               |
| 混合控制调度    | 控制器决定何时使用 sync / async 执行节点 | 主训练器、配置中控制                                    |
| 3D 重分布引擎  | 高效切换 rollout / 训练 模型状态      | `sharding_manager/*.py`                       |
| 后端解耦      | Actor / Critic / RM 支持多种框架  | `dp_*.py`, `megatron_*.py`, `vllm_rollout.py` |
| 模型 / 数据并行 | 用 Megatron / FSDP 做透明并行     | `fsdp_workers.py`, `megatron_workers.py`      |

# WorkerGroup
## 1. 🏗 WorkerGroup 的构建方式

* 每个 `WorkerGroup` 管理一组远程运行的 `Worker` 实例，每个 Worker 通常部署在一张 GPU 上，负责具体计算任务，构造过程即在 controller 进程中启动整个组 ([verl.readthedocs.io][1])。
* `WorkerGroup` 本质上是 controller 与多 GPU worker 的代理，它负责将 controller 调用拆解为多 GPU 并行远程调用，包括：**分割输入、分发、异步执行、结果收集、拼接** ([verl.readthedocs.io][2])。
* PPO 中定义了三种主要 WorkerGroup：

  * **ActorRolloutRef**：封装 actor 模型、rollout 推理、reference 策略计算，可组合成不同角色以最大复用代码，借助 create\_colocated\_worker\_cls 创建共享 GPU 的 Ray actor 类，便于快速权重迁移、实现 LoRA PPO 架构 ([verl.readthedocs.io][1])。
  * **Critic**：负责值函数估计。
  * **Reward**：负责奖励模型计算。

这些组的构建都通过 Ray 的资源池（ResourcePool）完成，将 worker 和 GPU 资源绑定在一起 ([verl.readthedocs.io][1])。

---

## 2. 🧩 Worker 方法定义与装饰器机制

以 `ActorRolloutRefWorker` 为例：

* 它定义了一系列对 controller 可见的接口（remote RPC 方法），如：

  * `init_model()`
  * `generate_sequences(DataProto)`
  * `compute_log_prob(...)`
  * `compute_ref_log_prob(...)`
  * `update_actor(...)`
  * `save_checkpoint()` 等 ([verl.readthedocs.io][1])。

* 这些方法通过 `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` 装饰器标记，明确指出：

  * 如何切分数据（dispatch\_mode）
  * 阻塞行为、执行模式等 metadata
  * 装饰器本身封装输入输出元数据，但不改变本体逻辑 ([verl.readthedocs.io][2])。

* 在 `WorkerGroup` 初始化时，会自动扫描 Worker 类中所有带有 `@register` 的方法，并绑定为 `WorkerGroup` 的方法：

  * 读取 dispatch\_mode，查表获取对应的 dispatch\_fn 和 collect\_fn（例如 DP\_COMPUTE\_PROTO 显式分片、收集逻辑）；
  * 获取 execute\_fn，根据 execute\_mode 决定同步异步行为；
  * 最终生成带分发、收集、并发执行封装的 Group 方法 ([verl.readthedocs.io][2])。

这样 controller 只需调用：

```python
output = actor_rollout_ref_wg.generate_sequences(data)
```

就触发：数据分割 → 分发至各 `Worker` → 并发 remote 调用 → 结果 ray.get、拼接返回。无需手动写循环逻辑 ([verl.readthedocs.io][1])。

---

# PPO 主循环

借助上述 WorkerGroup，PPO 主控流程可以像单进程那样简洁书写各阶段：

```python
for prompt in dataloader:
    output = actor_rollout_ref_wg.generate_sequences(prompt)
    old_lp = actor_rollout_ref_wg.compute_log_prob(output)
    ref_lp = actor_rollout_ref_wg.compute_ref_log_prob(output)
    values = critic_wg.compute_values(output)
    rewards = reward_wg.compute_scores(output)
    advantages = compute_advantages(values, rewards)
    merged = output.union(old_lp, ref_lp, values, rewards, advantages)
    actor_rollout_ref_wg.update_actor(merged)
    critic_wg.update_critic(merged)
```
generate_sequences、compute_log_prob、compute_ref_log_prob、compute_values、compute_scores、update_actor、update_critic 均是 WorkerGroup 的方法，这些由装饰器（如 @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)）封装后，自动执行：输入切片、数据分发到 GPU worker、远程调用并发计算、结果收集拼接返回等流程
control loop 看起来像普通的流程控制，但实际执行时每一步都调用远程 GPU worker，controller 自身并不做这些计算，只负责调度和接收返回。
compute_advantages 是在 controller 本地执行的小规模计算，不会涉及 GPU，只是简单地用返回的 values 和 rewards 计算 advantage。
这些调用背后都由 WorkerGroup 分发执行细节封装，controller 无需显式处理多 GPU 的分发与收集逻辑 ([verl.readthedocs.io][1])。

## 4. 为什么它“像单进程”——HybridFlow 的精髓
HybridFlow 的核心设计理念是：用单控制器（Single‑Controller）形式编写训练控制流程，而实际执行过程则由多进程（多 GPU workers）完成。
这样一来，用户只需编写像单机训练那样的 PPO 控制流程；
框架负责调度分布式执行，使得 backend 可在 FSDP、Megatron、vLLM 等之间自由切换；
控制逻辑和计算逻辑完全解耦，控制流程对 backend 完全透明

---

## ✅ 核心要点总结

| 概念                          | 说明                                                                                |
| --------------------------- | --------------------------------------------------------------------------------- |
| `WorkerGroup` 构造            | 初始化期间绑定多个远程 Worker，并为其注入带 dispatch/collect 逻辑的方法接口                                |
| `@register`                 | 装饰 Worker 中的方法，指定 dispatch 模式（如 DP\_COMPUTE\_PROTO）和执行特性                          |
| `Dispatch.DP_COMPUTE_PROTO` | 按数据并行分片输入，远程调用各 Worker，再收集拼接输出                                                    |
| 控制器流程                       | 主训练循环保持简洁，将复杂的多卡异构执行隐藏于 WorkerGroup 封装中                                           |
| 灵活部署                        | 修改 ResourcePool 或 WorkerGroup 构成即可切换 backend（Megatron / FSDP / vLLM 等），无需触及控制流程逻辑 |

[1]: https://verl.readthedocs.io/en/latest/hybrid_flow.html?utm_source=chatgpt.com "HybridFlow Programming Guide - verl documentation - Read the Docs"
[2]: https://verl.readthedocs.io/en/latest/single_controller.html?utm_source=chatgpt.com "The Design of verl.single_controller - verl documentation"


# Reward
https://github.com/volcengine/verl/tree/main/verl/utils/reward_score
https://verl.readthedocs.io/en/latest/preparation/reward_function.html
对于每个数据集，我们需要实现奖励函数或利用奖励模型来计算生成响应的奖励
我们支持 GSM8k 和 MATH 数据集的奖励函数。对于 RLHF 数据集（例如 full_hh_rlhf）和代码生成（例如 APPS），我们分别使用奖励模型和 SandBox（即将开源）进行评估。
在 PPO 训练后脚本 main_ppo.py 的入口点中，我们实现了一个 RewardManager，它利用预先实现的奖励函数来计算每个响应的分数。
在 RewardManager 中，我们实现了一个 __call__ 函数来计算每个响应的分数。所有奖励函数均由 compute_score_fn 执行。输入是一个 DataProto，其中包括：
input_ids、attention_mask：应用 chat_template 后的 input_ids 和 attention_mask，包括提示和响应
响应 ：响应标记
ground_truth：当前提示的基本事实字符串。存储在 DataProto 中的 non_tensor_batch，应在 parquet 文件中进行预处理。
data_source：当前提示的数据集名称。存储在 non_tensor_batch DataProto 中，应在 parquet 文件中进行预处理。 对响应进行detokenize后，响应字符串和基本事实字符串将输入到 compute_score_fn 中，以计算每个响应的分数。

## 一、主要模块对比

### ✅ `default_compute_score`（已替代 `_default_compute_score`）

* **功能**：通用策略，用于常规 RLHF 数据集，尤其当没有专用函数时的默认处理方式。
* **备注**：旧版函数 `_default_compute_score` 已被弃用，官方建议使用 `verl.utils.reward_score.default_compute_score` ([GitHub][1])。

### 📊 `gsm8k.py`

* **用途**：针对 GSM8k 数学题数据集的规则计算方式。
* **实现方式**：

  * 要求模型将答案放在格式 `#### 答案` 中；
  * **完全正确** → reward = 1；
  * **格式正确但答案错误** → reward = 0.1；
  * **格式错误** → reward = 0 ([verl.readthedocs.io][2], [Hugging Face][3])。

### 📐 `math.py`

* **用途**：处理 MATH 数据集（复杂数学推理）。
* **实现方式**：参考 lm-evaluation-harness 的 Hendrycks 方法，实现分步推理正确度衡量和得分评估 ([verl.readthedocs.io][2], [Hugging Face][3])。

### 💻 `prime_code`（代码生成任务）

* **用途**：针对生成代码正确性的评估。
* **实现方式**：

  * 首先对所有测试用例运行 correctness 检查。
  * 如果全部通过，则直接返回成功；
  * 若失败，则对前 10 测试用例逐个判断正确率（continuous 模式下始终检查前 10） ([GitHub][4])。

---

## 二、主要区别一览

| 模块                      | 应用场景           | 计算方式            | 得分策略         |
| ----------------------- | -------------- | --------------- | ------------ |
| `default_compute_score` | 通用 RLHF 或自定义任务 | 使用默认分值策略        | 较通用，但较通用化    |
| `gsm8k.py`              | GSM8k 数学题      | 字符串匹配格式 + 答案对比  | 1／0.1／0 三档评分 |
| `math.py`               | MATH 数据集       | 按解题步骤逻辑评估正确性    | 连续分值，看推理质量   |
| `prime_code`            | 代码生成／编程任务      | 测试用例正确性和连续性分步检测 | 全对即满分，部分对按比例 |
