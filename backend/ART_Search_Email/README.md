# 单Agent多工具训练

## 准备数据
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2
python train_email_search_agent.py build-db

# 训练
python train_email_search_agent.py train

## kill掉进程
ps aux | grep train_email_search_agent.py | grep -v grep | awk '{print $2}' | xargs kill -9
