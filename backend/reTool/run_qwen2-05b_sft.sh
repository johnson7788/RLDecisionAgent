#!/bin/bash
set -x
# 使用哪个显卡
export CUDA_VISIBLE_DEVICES=1,2

nnodes=1
nproc_per_node=2

experiment_name=multiturn-sft-Qwen2.5-0.5B-Instruct
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

TRAIN_DATA=$DATA_ROOT/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
EVAL_DATA=$DATA_ROOT/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
MODEL_PATH=$HDFS_ROOT/model/Qwen2.5-0.5B-Instruct
SAVE_PATH=$DATA_ROOT/checkpoint/$experiment_name

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