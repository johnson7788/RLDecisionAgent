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
