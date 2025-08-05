# 使用aliyun运行(运行环境Tesla V100-SXM2-32GB)
## 安装docker
```
https://docs.docker.com/engine/install/ubuntu/

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

# 安装GPU支持
```
安装
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

配置docker
sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

验证:
docker info | grep -i runtime
 Runtimes: runc io.containerd.runc.v2 nvidia
 Default Runtime: runc

sudo docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi
```

## Docker镜像
```
docker pull vemlp-boe-cn-beijing.cr.volces.com/preset-images/verl:v0.4.1
挂载时区
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name verl vemlp-boe-cn-beijing.cr.volces.com/preset-images/verl:v0.4.1 sleep infinity
docker start verl
docker exec -it verl bash

# 镜像中的verl 0.41版本 测试
cd code/verl
pip install --no-deps -e .

# 验证GPU
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'
------>
True
1

# pip freeze | grep sglang
sglang==0.4.6.post5
# pip freeze | grep vllm
vllm==0.8.5.post1


```

## 训练char_count
### SFT步骤
```
创建数据:
python create_dataset.py
ls -lR ~/data/char_count
/root/data/char_count:
total 8
drwxr-xr-x 2 root root 4096 Aug  5 21:26 rl
drwxr-xr-x 2 root root 4096 Aug  5 21:26 sft

/root/data/char_count/rl:
total 740
-rw-r--r-- 1 root root  79844 Aug  5 21:26 test.parquet
-rw-r--r-- 1 root root 675332 Aug  5 21:26 train.parquet

/root/data/char_count/sft:
total 728
-rw-r--r-- 1 root root  76602 Aug  5 21:26 test.parquet
-rw-r--r-- 1 root root 667050 Aug  5 21:26 train.parquet

训练:
export HF_ENDPOINT=https://hf-mirror.com
bash train_sft.sh
输出:

```


### GRPO步骤













# 手动安装方式
### 阿里云GPU环境搭建
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


### Debug模式
```
配置Pycharm的环境
export CUDA_VISIBLE_DEVICES=1,2

配置运行命令和参数
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


### Document
```
nproc_per_node=8  # 单个机器的卡数量
trainer:
  default_local_dir: verl_sft/Qwen25_7b_sft  # 训练后的模型的保存路径
```