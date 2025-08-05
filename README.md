# 强化学习训练的决策型Agent(运行环境 3*4090显卡)
## 开发中(developing)

## Docker镜像
```
docker pull vemlp-boe-cn-beijing.cr.volces.com/preset-images/verl:v0.4.1
```

## 创建docker镜像， 使用 --net=host 后，容器会共享宿主机的网络栈，不需要再次进行端口映射了。
```
#docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl vemlp-boe-cn-beijing.cr.volces.com/preset-images/verl:v0.4.1 sleep infinity
挂载时区
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name verl vemlp-boe-cn-beijing.cr.volces.com/preset-images/verl:v0.4.1 sleep infinity
docker start verl
docker exec -it verl bash

# 验证GPU
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'
------>
True
1

```

| 参数                     | 含义                                                    |
| ---------------------- | ----------------------------------------------------- |
| `docker create`        | 创建一个容器（不会自动运行）                                        |
| `--runtime=nvidia`     | 指定使用 `nvidia` runtime，以便容器内可以访问 GPU（NVIDIA Docker 支持） |
| `--gpus all`           | 分配所有 GPU 给容器                                          |
| `--net=host`           | 容器使用主机网络（提高网络性能，也方便调试端口）                              |
| `--shm-size="10g"`     | 增加共享内存大小，默认只有 64MB，这里设置为 10GB（常用于 PyTorch/DL 框架防止报错）  |
| `--cap-add=SYS_ADMIN`  | 提升容器权限（比如运行一些需要额外权限的操作）                               |
| `-v .:/workspace/verl` | 挂载当前目录（主机）到容器内 `/workspace/verl`                      |
| `--name verl`          | 给容器命名为 `verl`                                         |
| `vemlp-...:v0.4.1`     | 使用的基础镜像，来自某个私有镜像仓库                                    |
| `sleep infinity`       | 启动命令：容器运行一个不会退出的任务（无限睡眠），保持容器活着但不干别的                  |

docker start verl
启动名为 verl 的容器（前面已经创建了，但还未运行）。

## 安装一些依赖
```
cd /workspace/verl
pip3 install --no-deps -e .

pip freeze | grep verl
# Editable Git install with no remote (verl==0.5.0.dev0)
-e /workspace/verl/verl

镜像中已经安装sglang和vllm
# pip freeze | grep sglang
sglang==0.4.6.post5
# pip freeze | grep vllm
vllm==0.8.5.post1

不用再次安装这些依赖
pip3 install -e .[vllm]
pip3 install -e .[sglang]
```

| 命令                           | 是否安装依赖 | 是否可选安装         | 开发模式 | 典型场景              |
| ---------------------------- | ------ | -------------- | ---- | ----------------- |
| `pip install --no-deps -e .` | ❌ 不装依赖 | ❌ 不带扩展         | ✅ 是  | 你自己控制依赖，比如用 conda |
| `pip install -e .[vllm]`     | ✅ 安装依赖 | ✅ 安装 vllm 扩展   | ✅ 是  | 你要使用 vllm 功能      |
| `pip install -e .[sglang]`   | ✅ 安装依赖 | ✅ 安装 sglang 扩展 | ✅ 是  | 你要使用 sglang 的部分功能 |
