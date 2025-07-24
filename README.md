# 强化学习训练的决策型Agent


# 阿里云GPU环境搭建
```
pip install virtualenv
virtualenv -p python3.12 venv
source venv/bin/activate

#torch和torchvision版本要一致, 注意cuda版本
https://pytorch.org/get-started/previous-versions/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 稳定安装flash-attn方法
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
pip install torch packaging ninja
# 2.7.3编译速度快，还不报错，如果2.8.1报错
pip install -U flash-attn==2.7.3 --no-build-isolation

#方法2:
# flash_attn的版本一定要和torch，cuda，python的版本一致
https://github.com/Dao-AILab/flash-attention/releases
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install  --no-deps flash_attn-2.8.0.post2+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 验证下面是否报错，直到不报错
python -c "from transformers import PreTrainedModel; print('Imported successfully')"

```

## Debug模式
```
# Script模式
/home/wac/johnson/anaconda3/envs/gpt/bin/torchrun
# 参数
--standalone
--nnodes=1
--nproc_per_node=1
-m
verl.trainer.fsdp_sft_trainer
data.train_files=/home/wac/johnson/data/char_count/sft/train.parquet
data.val_files=/home/wac/johnson/data/char_count/sft/test.parquet
data.prompt_key=prompt
data.response_key=response
data.micro_batch_size_per_gpu=8
data.max_length=256
data.train_batch_size=256
use_remove_padding=True
model.partial_pretrain=HuggingFaceTB/SmolLM2-135M-Instruct
trainer.default_local_dir=./models/sft
trainer.project_name=char_count-sft
trainer.experiment_name=char_count-sft-SmolLM2-135M-Instruct
trainer.total_epochs=3
trainer.logger=console
```