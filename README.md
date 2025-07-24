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
pip install -U flash-attn==2.6.3 --no-build-isolation

#方法2:
# flash_attn的版本一定要和torch，cuda，python的版本一致
https://github.com/Dao-AILab/flash-attention/releases
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install  --no-deps flash_attn-2.8.0.post2+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn==2.6.3

# 验证下面是否报错，直到不报错
python -c "from transformers import PreTrainedModel; print('Imported successfully')"

```