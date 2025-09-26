# ms-swift还提供了基于Gradio的Web-UI界面及丰富的最佳实践。
🍎 模型类型：支持500+纯文本大模型、200+多模态大模型以及All-to-All全模态模型、序列分类模型、Embedding模型训练到部署全流程。
数据集类型：内置150+预训练、微调、人类对齐、多模态等各种类型的数据集，并支持自定义数据集。
硬件支持：CPU、RTX系列、T4/V100、A10/A100/H100、Ascend NPU、MPS等。
轻量训练：支持了LoRA、QLoRA、DoRA、LoRA+、ReFT、RS-LoRA、LLaMAPro、Adapter、GaLore、Q-Galore、LISA、UnSloth、Liger-Kernel等轻量微调方式。
分布式训练：支持分布式数据并行（DDP）、device_map简易模型并行、DeepSpeed ZeRO2 ZeRO3、FSDP、Megatron等分布式训练技术。
量化训练：支持对BNB、AWQ、GPTQ、AQLM、HQQ、EETQ量化模型进行训练。
🍊 RLHF训练：支持纯文本大模型和多模态大模型的DPO、GRPO、RM、PPO、GKD、KTO、CPO、SimPO、ORPO等人类对齐训练方法。
🍓 多模态训练：支持对图像、视频和语音不同模态模型进行训练，支持VQA、Caption、OCR、Grounding任务的训练。
🥥 Megatron并行技术：支持使用Megatron并行技术对CPT/SFT/DPO进行加速，现支持200+大语言模型。
界面训练：以界面的方式提供训练、推理、评测、量化的能力，完成大模型的全链路。
插件化与拓展：支持自定义模型和数据集拓展，支持对loss、metric、trainer、loss-scale、callback、optimizer等组件进行自定义。
🍉 工具箱能力：不仅提供大模型和多模态大模型的训练支持，还涵盖其推理、评测、量化和部署全流程。
推理加速：支持PyTorch、vLLM、SGLang和LmDeploy推理加速引擎，并提供OpenAI接口，为推理、部署和评测模块提供加速。
模型评测：以EvalScope作为评测后端，支持100+评测数据集对纯文本和多模态模型进行评测。
模型量化：支持AWQ、GPTQ、FP8和BNB的量化导出，导出的模型支持使用vLLM/SGLang/LmDeploy推理加速，并支持继续训练。


# 部署
https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT%E5%AE%89%E8%A3%85.html#id3
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.1

# 环境部署

##  尝试使用verl的镜像
```
# 获取镜像
docker pull modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.1
# 使用哪个GPU, 可以为all，或者某个显卡
docker create --runtime=nvidia --gpus "device=2" --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name swift modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.1 sleep infinity
# 启动容器
docker start swift
docker exec -it swift bash
```

## ~/.bashrc中配置使用的GPU和实用的hugging face镜像
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com

## 设置pip镜像源
```
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
# 克隆强化学习训练框架
```
cd verl-agent
pip install .
```

## 设置代理，安装git上的项目
```
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPs_PROXY=http://127.0.0.1:7890
pip install 'torchtune @ git+https://github.com/pytorch/torchtune.git'
pip install 'unsloth-zoo @ git+https://github.com/bradhilton/unsloth-zoo'
```

## WanDB docker 训练记录
```
docker pull wandb/local
docker run -d --restart always -v wandb:/vol -p 3005:8080 --name wandb-local wandb/local
#会提示您配置打开浏览器http://localhost:3005/authorize，新建一个本地普通用户, 粘贴key
输入邮箱和用户名创建一个本地的用户，得到类似这样的KEY， local-f2ca8cd44276ac92ca0a2c12641a6902beb6847d
粘贴到.env的环境变量配置文件中
```