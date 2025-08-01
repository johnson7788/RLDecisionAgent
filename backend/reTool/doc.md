# [run_qwen2-05b_sft.sh](run_qwen2-05b_sft.sh) 讲解

## 🚀 脚本总体上下文

该脚本使用 Verl 框架提供的 `verl.trainer.fsdp_sft_trainer` 模块，通过**PyTorch Fully Sharded Data Parallel (FSDP)** 安全分布式训练，实现对 Qwen‑2.5‑0.5B 模型的多轮对话 SFT 训练，数据来源于 ReTool‑SFT 的 parquet 格式数据集，该模型支持工具调用和多轮交互。Verl v0.4.x 版本已正式支持 Qwen‑2.5B、qwen3 及 multi‑turn SFT 功能 ([data.safetycli.com][1])。

---

## 🎯 脚本分解：每段参数含义

```bash
#!/bin/bash
set -x
# 使用哪个显卡
export CUDA_VISIBLE_DEVICES=1,2

nnodes=1
nproc_per_node=2
```

* `CUDA_VISIBLE_DEVICES=1,2`: 指定 GPU 卡编号为第 1 和第 2 张。
* `nnodes=1`、`nproc_per_node=2`: 表示单节点训练，每节点启动 2 个进程（2 GPUs），与上面的 `CUDA_VISIBLE_DEVICES` 对应。

```bash
experiment_name=multiturn-sft-Qwen2.5-0.5B-Instruct
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}
```

* `experiment_name`：此次实验命名，用于日志/模型文件组织。
* `HDFS_ROOT` 与 `DATA_ROOT`：分别用于指定模型存储、数据存放的根路径，若未设置则默认当前目录。

```bash
TRAIN_DATA=.../train‑00000‑of‑00001.parquet
EVAL_DATA=同上
MODEL_PATH=$HDFS_ROOT/model/Qwen2.5-0.5B-Instruct
SAVE_PATH=$DATA_ROOT/checkpoint/$experiment_name
```

* 指定训练与验证数据文件（Parquet 格式，多轮模型输入格式，包含 `messages` 和 `tools` 字段）；
* `model.partial_pretrain` 指向已有预训练模型路径；
* `SAVE_PATH` 是训练 checkpoints 的输出目录。

```bash
torchrun --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=16 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=wuxibin-multiturn-sft \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console"]' \
    trainer.total_epochs=2 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
```

### 参数功能总结：

#### **数据配置 (data.\*)**

* `data.train_files` & `data.val_files`: 指明训练和验证数据路径；
* `data.max_length=16384`: 支持的最大 token 长度，适用于长上下文；
* `data.train_batch_size=16`: 总 batch size；
* `data.micro_batch_size_per_gpu=2`: 每个 GPU 的微批量大小（16 = 2 GPUs × 2 micro-batch × gradient accumulation steps）；
* `data.multiturn.enable=true`: 启用多轮对话微调；
* `messages_key=messages`, `tools_key=tools`: 对应数据集中 multi‑turn 输入里用户/助手对话和工具调用字段 ([Docfork][2], [Hugging Face][3], [verl.readthedocs.io][4])。

#### **模型配置**

* `model.partial_pretrain=$MODEL_PATH`: 指向预训练模型 checkpoint，SFT 在此基础上 fine-tune；
* `model.strategy=fsdp`: 使用 FSDP 训练策略（分布式 shard）([GitHub][5], [data.safetycli.com][1])。

#### **训练器配置 (trainer.\*)**

* `trainer.default_local_dir=$SAVE_PATH`: checkpoint/log 保存路径；
* `trainer.project_name`, `trainer.experiment_name`: 分别用于日志系统（如 wandb）标识；
* `trainer.logger='["console"]'`: 只在控制台输出日志；
* `trainer.total_epochs=2`: 共训练 2 个 epoch。

#### **性能与 parallelism**

* `ulysses_sequence_parallel_size=2`: 启用 Ulysses sequence parallelism，支持长 context 分段并行训练（Qwen‑系列支持此方法）([data.safetycli.com][1], [Hugging Face][3], [GitHub][6])。
* `use_remove_padding=true`: 去除 padding token，进一步节约显存并加速计算。

---

## 🔍 为什么这些设置组合起来？

1. **FSDP + Ulysses sequence parallelism** 让模型在 GPU 数量有限时依然训练长上下文对话模型。
2. **Multi-turn 数据格式**：管理对话历史和工具调用，符合 SGLang / ReTool 的训练需求（在 Verl 新版本支持 multi‑turn 模型训练）([data.safetycli.com][1])。



---

## 🧠 总结

该脚本是一个标准的使用 Verl FSDP (v0.4.x) + dual‑GPU + sequence parallelism + multi‑turn 数据结构进行 SFT 微调的示例。核心功能包括：

* 从 Parquet 格式的数据里加载多轮对话样本；
* 支持工具调用上下文（`tools_key` 字段）；
* 借助 FSDP 和 Ulysses sequence parallelism，有效处理长上下文与大模型；
* 控制 GPU 所有可见设备、微批大小和总 Epoch，使训练资源可控。

如果你还想调整例如 mixed‑precision、wandb logging、或者自定义 prompt/response key，Sc配置 yaml 文件或者命令行参数均可轻松扩展 ([Hugging Face][7])。


# DAPO训练脚本 [run_qwen2-05b_dapo.sh](run_qwen2-05b_dapo.sh)
我帮你整体梳理一下这个训练脚本的逻辑结构和参数，最后指出可能存在的问题或潜在风险。

---

## 脚本结构和参数说明

### 1. 环境变量和数据路径

```bash
export CUDA_VISIBLE_DEVICES=1,2
```

指定使用第2和第3个GPU（从0开始编号）。

```bash
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

dapo_math_17k=$DATA_ROOT/dataset/BytedTsinghua/train
aime_2024=$DATA_ROOT/dataset/Maxwell/validation
model_path=$HDFS_ROOT/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/huggingface

train_files="['$dapo_math_17k']"
test_files="['$aime_2024']"
```

* 数据和模型路径设定，带默认值（当前目录）。
* 训练集和测试集路径通过字符串形式传递，注意是 `train_files="['path']"`，是字符串，看调用程序是否能正确解析。

---

### 2. wandb和工具配置

```bash
tool_config_path=recipe/retool/sandbox_fusion_tool_config.yaml
project_name=wuxibin_retool
experiment_name=qwen2.5-05b_dapo
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name
```

wandb项目名和实验名，日志保存目录。

---

### 3. 算法相关参数

```bash
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=8
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6

n_resp_per_prompt=16
n_resp_per_prompt_val=30
```

* 用了 `grpo` 作为advantage估计器。
* KL相关的权重都设置为0，且不开启KL loss和KL奖励。
* 剪切比率范围和学习率等。
* 对话最大轮数8，prompt和response长度都挺大（尤其是response 16384token）。
* 生成多个响应数量。

---

### 4. 性能相关

```bash
infer_tp=4 # vllm 推理tensor model parallel
train_sp=8 # 训练并行度
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))
```

* 训练与推理的并行度配置。
* offload开关，可能控制FSDP参数和优化器offload。
* 计算最大token长度（提示+回复）和log prob最大token长度。

---

### 5. 执行python训练模块，并传递大量参数

调用命令非常长，主要包括：

* algorithm相关参数（adv\_estimator，KL等）
* data相关参数（train/test文件，batch size，prompt/response最大长度，数据过滤和截断方式，自定义数据集和奖励函数）
* actor模型和优化参数（模型路径，clip ratio，学习率，ppo mini batch大小，token长度，fsdp offload等）
* rollout相关（vllm，异步模式，tensor并行，multi-turn设置，tool config，top-p，温度，响应数）
* trainer相关（日志，GPU数量，验证频率，存储路径，训练epoch数等）

---

## 可能的问题和建议

1. **train\_files 和 test\_files 格式**
   你传的是字符串形式的 `train_files="['$dapo_math_17k']"`，如果程序里直接用这个字符串，可能会识别成一个单字符串，而不是列表。
   **建议：** 确认 `verl.trainer.main_ppo` 的代码中对这类参数的解析逻辑，如果是用 `eval` 或 `json.loads`解析字符串为list则没问题，否则建议传成JSON格式或直接不加引号的列表格式。

2. **response最大长度 16384 token过大**
   16k token的响应长度非常大，训练和推理会非常消耗显存，且可能导致OOM或显存溢出。确认你的GPU显存是否足够支持这个长度。
   **建议：** 如果显存不足，考虑缩短max\_response\_length。

3. **offload设为True**
   你使用了FSDP参数和优化器参数的offload，确保你的硬件和依赖版本支持这个功能，否则可能导致训练不稳定。

4. **clip\_ratio\_low 和 clip\_ratio\_high 设置**
   通常PPO的clip ratio设置区间大概在0.1\~0.3之间，你设置的是0.2到0.28，基本合理，但有点偏窄。clip\_ratio\_c=10.0比较大，确认代码里clip\_ratio\_c的含义。

5. **`data.truncation='error'`**
   这个参数很关键，表示遇到超长输入时抛错。如果你的数据中有超过max\_prompt\_length的输入，会导致程序直接报错退出。
   **建议：** 如果数据不干净，建议改成`truncate`或`ignore`，或者确认数据清洗。

6. **n\_resp\_per\_prompt 和 n\_resp\_per\_prompt\_val**
   生成16和30个响应，对于计算资源和推理时间要求比较高，确认你的训练和推理环境支持。

7. **`use_kl_loss=False`和`kl_loss_coef=0.0`，`use_kl_in_reward=False`，`kl_coef=0.0`**
   如果你本想用KL来控制策略偏离基模型，现在全部关闭了，确认是否符合你的训练目标。

8. **训练epoch只有1**
   `trainer.total_epochs=1`，表示只训练一轮，是否符合预期？

9. **CUDA\_VISIBLE\_DEVICES**
   设置为`1,2`，但`trainer.n_gpus_per_node=2`，符合设置，但要确保机器上第1和第2号GPU状态良好。


## SFT的模型输出结果，2张显卡
```
ls -R checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/
checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/:
data.pt                             fsdp_config.json              model_world_size_2_rank_1.pt
extra_state_world_size_2_rank_0.pt  huggingface                   optim_world_size_2_rank_0.pt
extra_state_world_size_2_rank_1.pt  model_world_size_2_rank_0.pt  optim_world_size_2_rank_1.pt

checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/huggingface:
added_tokens.json  config.json  generation_config.json  merges.txt  special_tokens_map.json  tokenizer.json  tokenizer_config.json  vocab.json

```