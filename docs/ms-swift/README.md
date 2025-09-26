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


## SFT训练
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot

## SFT推理， 推理慢，但是加载速度快
swift infer \
    --adapters ./output/v0-20250926-114832/checkpoint-94 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```
[INFO:swift] Input `exit` or `quit` to exit the conversation.
[INFO:swift] Input `multi-line` to switch to multi-line input mode.
[INFO:swift] Input `reset-system` to reset the system and clear the history.
[INFO:swift] Input `clear` to clear the history.
<<<
<<<
<<< 你是谁
我是swift-robot，由swift开发的人工智能聊天机器人。我被设计用来理解和生成自然语言文本，以便与人类进行交流和对话。如果您有任何问题或需要帮助，请随时告诉我！
--------------------------------------------------
```

## 合并lora的权重，然后使用vllm推理， 推理速度更快，但是加载时慢，运行结束后会生成merged的模型权重
例如：output/v0-20250926-114832/checkpoint-94-merged/
swift infer \
    --adapters ./output/v0-20250926-114832/checkpoint-94 \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048


## 启动WebUI,然后使用访问: http://xxxx:7860/
swift web-ui --lang zh

## 测试的WebUI, 然后使用访问: http://xxxx:7860/
swift app --model Qwen/Qwen2.5-7B-Instruct --adapters ./output/v0-20250926-114832/checkpoint-94 --stream true

# 部署（微调后模型）
使用以下命令启动部署服务端。如果权重使用全参数训练，请使用--model替代--adapters指定训练的checkpoint目录。你可以参考推理和部署文档介绍的客户端调用方式：curl、openai库和swift客户端进行调用。

swift deploy \
    --adapters ./output/v0-20250926-114832/checkpoint-94 \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048 \
    --served_model_name 'mylora-model'
这里将给出使用vLLM对多LoRA进行部署并调用的完整例子。

vllm的服务端
首先你需要安装vLLM：pip install vllm -U，并在部署时使用--infer_backend vllm，这通常可以显著加速推理速度。

我们预先训练了2个基模型为Qwen/Qwen2.5-7B-Instruct的不同自我认知LoRA增量权重（可以直接跑通），我们可以在args.json中找到相关信息。你需要在部署时修改--adapters指定训练好的LoRA权重本地路径即可。

swift deploy \
    --adapters lora1=swift/test_lora lora2=swift/test_lora2 \
    --infer_backend vllm \
    --temperature 0 \
    --max_new_tokens 2048

客户端
这里只介绍使用openai库进行调用。使用curl、swift客户端调用的例子可以参考推理和部署文档。

```
from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url=f'http://127.0.0.1:8000/v1',
)
models = [model.id for model in client.models.list().data]
print(f'models: {models}')

query = 'who are you?'
messages = [{'role': 'user', 'content': query}]

resp = client.chat.completions.create(model=models[1], messages=messages, max_tokens=512, temperature=0)
query = messages[0]['content']
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

gen = client.chat.completions.create(model=models[2], messages=messages, stream=True, temperature=0)
print(f'query: {query}\nresponse: ', end='')
for chunk in gen:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
"""
models: ['Qwen2.5-7B-Instruct', 'lora1', 'lora2']
query: who are you?
response: I am an artificial intelligence model named swift-robot, developed by swift. I can answer your questions, provide information, and engage in conversation. If you have any inquiries or need assistance, feel free to ask me at any time.
query: who are you?
response: I am an artificial intelligence model named Xiao Huang, developed by ModelScope. I can answer your questions, provide information, and engage in conversation. If you have any inquiries or need assistance, feel free to ask me at any time.
"""
```