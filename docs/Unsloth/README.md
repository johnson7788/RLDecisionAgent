# 环境准备
llama factory 容器的的trl==0.9.6
安装：docs/llamafactory/README.md
pip install -r requirements.txt

# SFT代码
[unsloth_sft.py](unsloth_sft.py)
## SFT训练
python unsloth_sft.py
使用的模型是： unsloth/Qwen3-4B-Instruct-2507

# Thinking模型训练
[unsloth_thinking.py](unsloth_thinking.py)
## Thinking模型训练
python unsloth_thinking.py
使用的模型是： unsloth/Qwen3-4B-Thinking-2507

# GRPO强化学习训练， 需要安装vllm
[unsloth_GRPO.py](unsloth_GRPO.py)
## GRPO模型训练
python unsloth_GRPO.py
使用的模型是： unsloth/Qwen3-4B-Base
