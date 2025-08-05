# 使用强化学习方法（GRPO 和 PPO）训练大语言模型（Qwen2.5-32B），在数学推理任务中**策略性地使用工具**，以提升解题准确率。

# 文档
https://www.notion.so/verl-reTool-recipe-2398b5b7feba80a58156fa936f9f8de6

# 1. 数据下载(这里用于强化学习训练的数据集)

下载和处理强化学习训练数据BytedTsinghua-SIA/DAPO-Math-17k
```
export HF_ENDPOINT=https://hf-mirror.com
python3 dapo_multiturn_w_tool.py
```

```
输出
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████| 145/145 [00:00<00:00, 518kB/s]
dapo-math-17k.parquet: 100%|███████████████████████████████████████████████████████████████████████▉| 299M/299M [03:21<00:00, 1.48MB/s]
Generating train split: 1791700 examples [00:03, 449608.47 examples/s]
Map:  19%|██████████████▍                                                            | 345877/1791700 [00:46<03:03, 7857.80 examples/s]

```
数据会下载到./dataset
```
ls ./dataset
train.parquet
```


数据条数: 1791700
列名： {'train': ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']}

具体数据, 主要用于数学题解任务的微调或评估。
prompt: 数学题目正文 + 回答格式要求
ability: MATH,纯数学能力
reward_model: 例如"reward_model": {
    "ground_truth": "34",
    "style": "rule-lighteval/MATH_v2"
}
 正确答案

```
Sample 0:
{'data_source': 'math_dapo', 'prompt': [{'content': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after "Answer:".', 'role': 'user'}], 'ability': 'MATH', 'reward_model': {'ground_truth': '34', 'style': 'rule-lighteval/MATH_v2'}, 'extra_info': {'index': '9a9b6eb4-a1cb-49d1-8c1e-62eaf2f74079', 'need_tools_kwargs': True, 'tools_kwargs': {'code_interpreter': {'create_kwargs': {'ground_truth': '34'}}}}}

Sample 1:
{'data_source': 'math_dapo', 'prompt': [{'content': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nLet $ABCD$ be a unit square in the plane. Points $X$ and $Y$ are chosen independently and uniformly at random on the perimeter of $ABCD$. If the expected value of the area of triangle $\\triangle AXY$ can be expressed as $\\frac{m}{n}$ for relatively prime positive integers $m$ and $n$, compute $m+n$.\n\nRemember to put your answer on its own line after "Answer:".', 'role': 'user'}], 'ability': 'MATH', 'reward_model': {'ground_truth': '113', 'style': 'rule-lighteval/MATH_v2'}, 'extra_info': {'index': 'b426d104-244d-4831-a2c4-cd756b61700a', 'need_tools_kwargs': True, 'tools_kwargs': {'code_interpreter': {'create_kwargs': {'ground_truth': '113'}}}}}

Sample 2:
{'data_source': 'math_dapo', 'prompt': [{'content': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nLet $a, b, c$ be distinct numbers such that the equations $x^2 + ax + 1 = 0$ and $x^2 + bx + c = 0$ have a common real root, and the equations $x^2 + x + a = 0$ and $x^2 + cx + b = 0$ also have a common real root. Compute the sum $a + b + c$.\n\nRemember to put your answer on its own line after "Answer:".', 'role': 'user'}], 'ability': 'MATH', 'reward_model': {'ground_truth': '-3', 'style': 'rule-lighteval/MATH_v2'}, 'extra_info': {'index': '6ff0b17f-7e5c-4ae9-b5e9-63ebecd2b9f7', 'need_tools_kwargs': True, 'tools_kwargs': {'code_interpreter': {'create_kwargs': {'ground_truth': '-3'}}}}}
```

基本结构
```
{
    "data_source": "math_dapo",
    "prompt": [  # 单轮用户问题
        {"content": "<数学题>", "role": "user"}
    ],
    "ability": "MATH",
    "reward_model": {
        "ground_truth": "<标准答案>",
        "style": "rule-lighteval/MATH_v2"
    },
    "extra_info": {
        "index": "<UUID>",
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "code_interpreter": {
                "create_kwargs": {
                    "ground_truth": "<标准答案>"
                }
            }
        }
    }
}
```

下载强化学习测试数据，"Maxwell-Jia/AIME_2024"
```
export HF_ENDPOINT=https://hf-mirror.com
python dapo_aime2024_data_process.py
```

## 📦 模型与数据

| 项目          | 内容                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------- |
| Base model  | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) —— 微调与 RL 都基于这个 LLM。                  |
| SFT dataset | [JoeYing/ReTool-SFT](https://huggingface.co/datasets/JoeYing/ReTool-SFT) —— 用于监督微调（SFT）。                            |
| RL dataset  | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) —— 用于强化学习（奖励建模）。 |
| Val dataset | [yentinglin/aime\_2025](https://huggingface.co/datasets/yentinglin/aime_2025) —— 用于评估模型的泛化能力。                       |

---

# 2. 下载模型
```
cd backend/reTool
python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='model/Qwen2.5-0.5B-Instruct')
```

## 🚀 微调阶段（SFT）

### 2.1 数据预处理(加载huggingface上的JoeYing/ReTool-SFT)

* 从 `ReTool-SFT` 数据集中提取训练样本，可能包含用户输入、tool-calling 格式、ground truth 等。
将 JoeYing/ReTool-SFT 数据集中原始的对话数据（包含 <code>、<interpreter>、<answer> 等标签）转换为标准的多轮工具调用格式（tool-calling messages）并存储为 .parquet 格式数据。
原始数据中每条样本是一个数学问题解决过程（由模型回答），包括：

用户提问

模型输出代码（带 <code>...</code> 标签）

工具执行返回结果（带 <interpreter>...</interpreter> 标签）

模型最终输出答案（带 <answer>...</answer> 标签）

目标是把这些内容拆分成标准的 message 格式，便于训练支持工具调用的语言模型。

需要配置Code Sandbox Agent，
https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment
Code Sandbox Agent 是火山引擎函数服务（veFaaS） 基于开源 SandboxFusion 项目，面向终端用户和 AI 套件开发者推出的 Sandbox 代码沙箱服务。用户输入编程任务，即可由豆包大模型分析编程任务，生成对应语言的程序代码，最后用户提交运行请求，Sandbox 运行代码并输出执行结果。整个使用过程可以看作为：大模型帮助解决编程问题并输出程序，Sandbox 帮助执行程序、验证代码正确性。
部署：
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
测试：
curl 'http://localhost:8080/run_code' \
  -H 'Content-Type: application/json' \
  --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
输出：
{
  "status": "Success",
  "message": "",
  "compile_result": null,
  "run_result": {
    "status": "Finished",
    "execution_time": 0.016735315322875977,
    "return_code": 0,
    "stdout": "Hello, world!\n",
    "stderr": ""
  },
  "executor_pod_name": null,
  "files": {}
}

### 2.2 SFT数据集生成, 注意更改sandbox_fusion_tool_config.yaml的配置中的sandbox_fusion_url
```bash
export HF_ENDPOINT=https://hf-mirror.com
python retool_sft_preprocess.py
```
原理:
```
读取工具配置文件 sandbox_fusion_tool_config.yaml，构造 tool schema。
加载 HuggingFace 数据集 JoeYing/ReTool-SFT。
对每个样本 row 执行 process 函数，提取标准格式的 messages 列表：
用户问题
助手的解释和代码（带 tool_calls）
工具执行结果（role=tool）
助手最后的答案
保存处理后的数据为 parquet 文件，用于训练或评估 downstream 模型。
最终的文件保存为当前目录下的：
wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
```
输出:
```
[process_output.md](process_output.md)

单条数据
{
  "messages": [
    {
      "content": "Find the greatest integer less than $\\sqrt{10} + \\sqrt{80}$.\n\nRemember to place the final answer in the last part using the format: \n\n\\boxed{{'The final answer goes here.'}}",
      "role": "user",
      "tool_calls": null
    },
    {
      "content": "Okay, so I need to find the greatest integer that's less than the sum of the square roots of 10 and 80. Let me think. Hmm, the problem is asking for the floor of √10 + √80. That means I have to calculate √10 and √80, add them together, and then take the integer part of that sum. Alright, let's break it down step by step.\n\nFirst, let's compute the approximate values of √10 and √80 using precise calculations.",
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "arguments": {
              "code": "import math\n\nsqrt10 = math.sqrt(10)\nsqrt80 = math.sqrt(80)\nsum_sqrt = sqrt10 + sqrt80\nprint(f\"√10 ≈ {sqrt10:.6f}, √80 ≈ {sqrt80:.6f}, sum ≈ {sum_sqrt:.6f}\")"
            },
            "name": "code_interpreter"
          },
          "type": "function"
        }
      ]
    },
    {
      "content": "√10 ≈ 3.162278, √80 ≈ 8.944272, sum ≈ 12.106550",
      "role": "tool",
      "tool_calls": null
    },
    {
      "content": "So, the sum of √10 and √80 is approximately 12.106550. This suggests the greatest integer less than this sum is 12. But to verify the precision, let's check the squares of the approximations to ensure they're accurate.",
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "arguments": {
              "code": "# Verify approximations by squaring\napprox_10 = 3.162278\napprox_80 = 8.944272\nprint(f\"(√10)^2 ≈ {approx_10**2:.6f}, (√80)^2 ≈ {approx_80**2:.6f}\")"
            },
            "name": "code_interpreter"
          },
          "type": "function"
        }
      ]
    },
    {
      "content": "(√10)^2 ≈ 10.000000, (√80)^2 ≈ 80.000001",
      "role": "tool",
      "tool_calls": null
    },
    {
      "content": "Both squares are extremely close to 10 and 80, confirming the approximations are accurate. Next, to ensure the sum isn't slightly over an integer threshold, let's compute more precise bounds.",
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "arguments": {
              "code": "# Calculate tighter bounds for √10 and √80\nlower_10 = 3.16227766017  # More precise lower bound for √10\nupper_10 = 3.16227766018  # Upper bound\nlower_80 = 8.94427190999   # Lower bound for √80\nupper_80 = 8.94427191000   # Upper bound\nlower_sum = lower_10 + lower_80\nupper_sum = upper_10 + upper_80\nprint(f\"Sum bounds: [{lower_sum:.6f}, {upper_sum:.6f}]\")"
            },
            "name": "code_interpreter"
          },
          "type": "function"
        }
      ]
    },
    {
      "content": "Sum bounds: [12.106549, 12.106549]",
      "role": "tool",
      "tool_calls": null
    },
    {
      "content": "The sum remains tightly around 12.106549, which is still less than 13. To further confirm, compute the exact square of the sum (√10 + √80):",
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "arguments": {
              "code": "# Calculate (√10 + √80)^2\nsum_squared = (sqrt10 + sqrt80) ** 2\nsqrt_sum_squared = math.sqrt(sum_squared)\nprint(f\"(√10 + √80)^2 ≈ {sum_squared:.6f}, sqrt of this ≈ {sqrt_sum_squared:.6f}\")"
            },
            "name": "code_interpreter"
          },
          "type": "function"
        }
      ]
    },
    {
      "content": "(√10 + √80)^2 ≈ 146.568542, sqrt of this ≈ 12.106550",
      "role": "tool",
      "tool_calls": null
    },
    {
      "content": "The squared sum is approximately 146.568542, and its square root confirms the original sum. Since 12.106550 is between 12 and 13, the greatest integer less than this value is 12.\n\nTherefore, the answer is:\n\n\n\\boxed{12}",
      "role": "assistant",
      "tool_calls": null
    }
  ],
  "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"code_interpreter\", \"description\": \"A tool for executing code.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"description\": \"The code to execute.\"}}, \"required\": [\"code\"]}}}]"
}
```

### 2. 启动SFT训练脚本， 共用了约 12 分钟多时间。

```bash
注意设置使用哪个显卡
export CUDA_VISIBLE_DEVICES=1,2
bash run_qwen2-05b_sft.sh
```
输出信息:
```
bash run_qwen2-05b_sft.sh
+ export CUDA_VISIBLE_DEVICES=1,2
+ CUDA_VISIBLE_DEVICES=1,2
+ nnodes=1
+ nproc_per_node=2
+ experiment_name=multiturn-sft-Qwen2.5-0.5B-Instruct
+ HDFS_ROOT=/workspace/verl/backend/reTool
+ DATA_ROOT=/workspace/verl/backend/reTool
+ TRAIN_DATA=/workspace/verl/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
+ EVAL_DATA=/workspace/verl/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
+ MODEL_PATH=/workspace/verl/backend/reTool/model/Qwen2.5-0.5B-Instruct
+ SAVE_PATH=/workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct
+ torchrun --nnodes=1 --nproc_per_node=2 -m verl.trainer.fsdp_sft_trainer data.train_files=/workspace/verl/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet data.val_files=/workspace/verl/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet data.max_length=16384 data.train_batch_size=32 data.multiturn.enable=true data.multiturn.messages_key=messages data.multiturn.tools_key=tools data.micro_batch_size_per_gpu=4 model.partial_pretrain=/workspace/verl/backend/reTool/model/Qwen2.5-0.5B-Instruct model.strategy=fsdp trainer.default_local_dir=/workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct trainer.project_name=wuxibin-multiturn-sft trainer.experiment_name=multiturn-sft-Qwen2.5-0.5B-Instruct 'trainer.logger=["console"]' trainer.total_epochs=6 ulysses_sequence_parallel_size=2 use_remove_padding=true
W0731 14:17:26.174000 2038 torch/distributed/run.py:792]
W0731 14:17:26.174000 2038 torch/distributed/run.py:792] *****************************************
W0731 14:17:26.174000 2038 torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0731 14:17:26.174000 2038 torch/distributed/run.py:792] *****************************************
Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
Skipping monkey patch for Qwen2ForCausalLM as use_fused_kernels is False or fused_kernels_backend is None
Normalize batch size by dp 1
Using sequence parallel size: 2
Using remove padding: True
Using SP rank 0 and size 1 for data distribution
Each SP rank gets different data, but the same data WITHIN the same rank
Using FSDP rank 0 and size 1 for data distribution
Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
Skipping monkey patch for Qwen2ForCausalLM as use_fused_kernels is False or fused_kernels_backend is None
functools.partial(<function _or_policy at 0x7ee4c8fe88b0>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ee4c8fe8790>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
NCCL version 2.21.5+cuda12.4
Number of steps/epoch 62, number of epochs 6, total number of steps 372
{'data': {'train_batch_size': 32, 'micro_batch_size': None, 'micro_batch_size_per_gpu': 4, 'train_files': '/workspace/verl/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet', 'val_files': '/workspace/verl/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet', 'prompt_key': 'question', 'response_key': 'answer', 'prompt_dict_keys': None, 'response_dict_keys': None, 'multiturn': {'enable': True, 'messages_key': 'messages', 'tools_key': 'tools', 'enable_thinking_key': 'enable_thinking'}, 'max_length': 16384, 'truncation': 'error', 'balance_dp_token': False, 'chat_template': None, 'custom_cls': {'path': None, 'name': None}, 'use_shm': False}, 'model': {'partial_pretrain': '/workspace/verl/backend/reTool/model/Qwen2.5-0.5B-Instruct', 'use_shm': False, 'fsdp_config': {'model_dtype': 'fp32', 'wrap_policy': {'min_num_params': 0}, 'cpu_offload': False, 'offload_params': False}, 'external_lib': None, 'enable_gradient_checkpointing': True, 'trust_remote_code': False, 'lora_rank': 0, 'lora_alpha': 16, 'target_modules': 'all-linear', 'use_liger': False, 'strategy': 'fsdp'}, 'optim': {'lr': 1e-05, 'betas': [0.9, 0.95], 'weight_decay': 0.01, 'warmup_steps_ratio': 0.1, 'clip_grad': 1.0, 'lr_scheduler': 'cosine'}, 'ulysses_sequence_parallel_size': 2, 'use_remove_padding': True, 'trainer': {'default_local_dir': '/workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct', 'default_hdfs_dir': None, 'project_name': 'wuxibin-multiturn-sft', 'experiment_name': 'multiturn-sft-Qwen2.5-0.5B-Instruct', 'total_epochs': 6, 'total_training_steps': None, 'logger': ['console'], 'seed': 1, 'save_freq': -1, 'test_freq': -1, 'nnodes': 1, 'n_gpus_per_node': 8, 'max_ckpt_to_keep': None, 'resume_mode': 'auto', 'resume_from_path': None, 'checkpoint': {'save_contents': ['model', 'optimizer', 'extra'], 'load_contents': '${trainer.checkpoint.save_contents}'}, 'device': 'cuda'}}
Epoch 1/6:   0%|                                                                                                | 0/62 [00:00<?, ?it/s]step:1 - train/loss:1.0605114698410034 - train/lr(1e-3):0.0002702702702702703
Epoch 1/6:   2%|█▍                                                                                      | 1/62 [00:09<09:36,  9.45s/it]step:2 - train/loss:1.0065364837646484 - train/lr(1e-3):0.0005405405405405405
Epoch 1/6:   3%|██▊                                                                                     | 2/62 [00:16<08:03,  8.06s/it]step:3 - train/loss:1.01118004322052 - train/lr(1e-3):0.0008108108108108109
Epoch 1/6:   5%|████▎                                                                                   | 3/62 [00:23<07:21,  7.48s/it]step:4 - train/loss:1.0045815706253052 - train/lr(1e-3):0.001081081081081081
Epoch 1/6:   6%|█████▋                                                                                  | 4/62 [00:31<07:27,  7.71s/it]step:5 - train/loss:0.9921297430992126 - train/lr(1e-3):0.0013513513513513514
Epoch 1/6:   8%|███████                                                                                 | 5/62 [00:38<07:01,  7.40s/it]step:6 - train/loss:0.8889124989509583 - train/lr(1e-3):0.0016216216216216218
Epoch 1/6:  10%|████████▌                                                                               | 6/62 [00:45<06:47,  7.27s/it]step:7 - train/loss:1.0498483180999756 - train/lr(1e-3):0.0018918918918918923
Epoch 1/6:  11%|█████████▉                                                                              | 7/62 [00:52<06:38,  7.24s/it]step:8 - train/loss:0.913988471031189 - train/lr(1e-3):0.002162162162162162
Epoch 1/6:  13%|███████████▎                                                                            | 8/62 [00:59<06:24,  7.11s/it]step:9 - train/loss:0.9712172150611877 - train/lr(1e-3):0.0024324324324324327
Epoch 1/6:  15%|████████████▊                                                                           | 9/62 [01:06<06:18,  7.13s/it]step:10 - train/loss:0.8903839588165283 - train/lr(1e-3):0.002702702702702703
Epoch 1/6:  16%|██████████████                                                                         | 10/62 [01:13<06:05,  7.03s/it]
step:246 - train/loss:0.5490185022354126 - train/lr(1e-3):7.79617909009489e-06
Epoch 2/2:  97%|██████████████████████████████████████████████████████████████████████████████████▎  | 121/125 [12:26<00:24,  6.06s/it]step:247 - train/loss:0.5956565737724304 - train/lr(1e-3):4.385849505708084e-06
Epoch 2/2:  98%|██████████████████████████████████████████████████████████████████████████████████▉  | 122/125 [12:32<00:18,  6.07s/it]step:248 - train/loss:0.6328058242797852 - train/lr(1e-3):1.949424798228239e-06
Epoch 2/2:  98%|███████████████████████████████████████████████████████████████████████████████████▋ | 123/125 [12:38<00:12,  6.01s/it]step:249 - train/loss:0.6089028716087341 - train/lr(1e-3):4.87379953478806e-07
Epoch 2/2:  99%|████████████████████████████████████████████████████████████████████████████████████▎| 124/125 [12:44<00:05,  5.99s/it]step:250 - train/loss:0.5836241245269775 - train/lr(1e-3):0.0
step:250 - val/loss:0.5834625363349915
Saving checkpoint to: /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250
[2025-07-31 14:50:14,935][/workspace/verl/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py][INFO] - [Rank 0] Saved model to /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/model_world_size_2_rank_0.pt
[2025-07-31 14:50:15,230][/workspace/verl/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py][INFO] - [Rank 1] Saved model to /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/model_world_size_2_rank_1.pt
[2025-07-31 14:50:18,218][/workspace/verl/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py][INFO] - [Rank 0] Saved optim to /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/optim_world_size_2_rank_0.pt
[2025-07-31 14:50:18,220][/workspace/verl/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py][INFO] - [Rank 0] Saved extra_state to /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/extra_state_world_size_2_rank_0.pt
[2025-07-31 14:50:18,393][/workspace/verl/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py][INFO] - [Rank 0] Saved model config and tokenizer class to /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/huggingface
[2025-07-31 14:50:18,876][/workspace/verl/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py][INFO] - [Rank 1] Saved optim to /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/optim_world_size_2_rank_1.pt
[2025-07-31 14:50:18,878][/workspace/verl/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py][INFO] - [Rank 1] Saved extra_state to /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/extra_state_world_size_2_rank_1.pt
Saved dataloader state to: /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/data.pt
Updated checkpoint tracker: /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/latest_checkpointed_iteration.txt
Final validation metrics: {'val/loss': 0.5834625363349915}
Epoch 2/2:  99%|████████████████████████████████████████████████████████████████████████████████████▎| 124/125 [16:00<00:07,  7.75s/it]
```
### 📉 训练和验证指标

| 步骤       | 指标           | 值                       |
| -------- | ------------ | ----------------------- |
| step 249 | `train/loss` | 0.6089                  |
| step 250 | `train/loss` | 0.5836                  |
| step 250 | `val/loss`   | **0.5835**              |
| step 250 | `lr`         | **0.0**（训练结束，学习率已衰减到 0） |

说明：

* **训练损失下降趋势明显**（从 0.6089 → 0.5836）。
* **验证损失非常接近训练损失**（0.5835 vs 0.5836），说明没有明显过拟合。

### 💾 Checkpoint 保存信息

以下文件被成功保存，表明训练结束后完整保存了模型状态：

| 保存对象               | 路径                                           | 说明                   |
| ------------------ | -------------------------------------------- | -------------------- |
| 模型参数（两卡）           | `model_world_size_2_rank_0.pt` / `rank_1.pt` | 分布式训练下的两个分片          |
| 优化器状态              | `optim_world_size_2_rank_{0,1}.pt`           | 保证断点训练恢复一致性          |
| FSDP 额外状态          | `extra_state_world_size_2_rank_{0,1}.pt`     | 主要是 scheduler、随机种子等  |
| Tokenizer 和 config | `huggingface/`                               | 方便导入为 Huggingface 格式 |
| dataloader 状态      | `data.pt`                                    | 保证断点恢复时的 batch 顺序等   |
| 最新检查点记录            | `latest_checkpointed_iteration.txt`          | 当前最新训练步数（step 250）   |


训练时显卡状态
```
nvidia-smi
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090 D      Off |   00000000:0D:00.0 Off |                  Off |
| 30%   58C    P2            160W /  425W |   22877MiB /  24564MiB |     90%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 4090 D      Off |   00000000:0E:00.0 Off |                  Off |
| 30%   60C    P2            159W /  425W |   22445MiB /  24564MiB |     94%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### ✅ 微调后评估结果(如何评估??)

```text
val-core/aime_2025/acc/mean@30: 0.24
val-aux/num_turns/mean: 7.2
```

* `acc/mean@30`: Top-30 validation accuracy（可能是 beam size=30）。
* `num_turns/mean`: 推理中平均轮数（每道题中模型调用工具的平均次数为 7.2）。

---

## 合并模型为huggingface格式
```
cd checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/
cp -a huggingface/* .
cd -
python /workspace/verl/verl/scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250 \
    --target_dir models/merged_sft_model
    
ls models/merged_sft_model
added_tokens.json  generation_config.json  model.safetensors        tokenizer.json         vocab.json
config.json        merges.txt              special_tokens_map.json  tokenizer_config.json
```

## 🔁 强化学习阶段（RL）

Retool 提供了两种 RL 策略：

### 🎯 GRPO（Generalized REINFORCE with Policy Optimization）

```bash
注意修改：model_path，即SFT的训练后的模型结果
bash run_qwen2-05b_dapo.sh
```

合并FSDP训练后的actor模型
```
检查最后一个step输出模型:
ls 
checkpoint/qwen2.5-05b_dapo/global_step_111

cd checkpoint/qwen2.5-05b_dapo/global_step_111/actor/
cp -a huggingface/* .
cd - 
python /workspace/verl/verl/scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoint/qwen2.5-05b_dapo/global_step_111/actor/ \
    --target_dir checkpoint/merged_dapo_model
输出:
Got device mesh tensor([0, 1], dtype=torch.int32), mesh_dim_names ('fsdp',)
Processing model shards with 2 (2,) in total
Loading 2 FSDP shards: 100%|█████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.29it/s]
Sliding Window Attention is enabled but not implemented for `eager`; unexpected results may be encountered.
Saving model to checkpoint/merged_dapo_model
Saving tokenizer to checkpoint/merged_dapo_model

ls -alht checkpoint/merged_dapo_model
total 1.2G
drwxr-xr-x 2 root root 4.0K Aug  5 11:50 .
-rw-r--r-- 1 root root  11M Aug  5 11:50 tokenizer.json
-rw-r--r-- 1 root root 1.6M Aug  5 11:50 merges.txt
-rw-r--r-- 1 root root 2.7M Aug  5 11:50 vocab.json
-rw-r--r-- 1 root root  605 Aug  5 11:50 added_tokens.json
-rw-r--r-- 1 root root  613 Aug  5 11:50 special_tokens_map.json
-rw-r--r-- 1 root root 7.2K Aug  5 11:50 tokenizer_config.json
-rw-r--r-- 1 root root 1.2G Aug  5 11:50 model.safetensors
-rw-r--r-- 1 root root  683 Aug  5 11:50 config.json
-rw-r--r-- 1 root root  242 Aug  5 11:50 generation_config.json
drwxr-xr-x 5 root root 4.0K Aug  5 11:50 ..
```

**评估结果（150步）**：

* acc\@30: **0.6**
* 平均调用轮数：**10**

说明 RL 后模型能更灵活使用工具，提升了准确率。

# 推理训练后的模型
```
# 使用哪个模型
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
ls checkpoint/merged_dapo_model
vllm serve checkpoint/merged_dapo_model --host 0.0.0.0 --port 5306
输出：
WARNING 08-05 11:56:20 [utils.py:2522] Methods determine_num_available_blocks,device_config,get_cache_block_size_bytes,initialize_cache not implemented in <vllm.v1.worker.gpu_worker.Worker object at 0x7b1f5d572410>
INFO 08-05 11:56:25 [parallel_state.py:1004] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0
INFO 08-05 11:56:25 [cuda.py:221] Using Flash Attention backend on V1 engine.
INFO 08-05 11:56:25 [topk_topp_sampler.py:44] Currently, FlashInfer top-p & top-k sampling sampler is disabled because FlashInfer>=v0.2.3 is not backward compatible. Falling back to the PyTorch-native implementation of top-p & top-k sampling.
INFO 08-05 11:56:25 [gpu_model_runner.py:1329] Starting to load model checkpoint/merged_dapo_model...
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.10it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.10it/s]

INFO 08-05 11:56:26 [loader.py:458] Loading weights took 0.27 seconds
INFO 08-05 11:56:26 [gpu_model_runner.py:1347] Model loading took 0.9271 GiB and 0.619115 seconds
INFO 08-05 11:56:36 [backends.py:420] Using cache directory: /root/.cache/vllm/torch_compile_cache/034dfb9f57/rank_0_0 for vLLM's torch.compile
INFO 08-05 11:56:36 [backends.py:430] Dynamo bytecode transform time: 9.21 s
INFO 08-05 11:56:39 [backends.py:136] Cache the graph of shape None for later use
INFO 08-05 11:57:02 [backends.py:148] Compiling a graph for general shape takes 26.11 s
INFO 08-05 11:57:10 [monitor.py:33] torch.compile takes 35.32 s in total
INFO 08-05 11:57:11 [kv_cache_utils.py:634] GPU KV cache size: 1,606,496 tokens
INFO 08-05 11:57:11 [kv_cache_utils.py:637] Maximum concurrency for 32,768 tokens per request: 49.03x
WARNING 08-05 11:57:34 [config.py:1239] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
INFO 08-05 11:57:34 [serving_chat.py:118] Using default chat sampling params from model: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
INFO 08-05 11:57:34 [serving_completion.py:61] Using default completion sampling params from model: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
INFO 08-05 11:57:34 [api_server.py:1090] Starting vLLM API server on http://0.0.0.0:5306
INFO 08-05 11:57:34 [launcher.py:28] Available routes are:
INFO 08-05 11:57:34 [launcher.py:36] Route: /openapi.json, Methods: GET, HEAD
INFO 08-05 11:57:34 [launcher.py:36] Route: /docs, Methods: GET, HEAD
INFO 08-05 11:57:34 [launcher.py:36] Route: /docs/oauth2-redirect, Methods: GET, HEAD
INFO 08-05 11:57:34 [launcher.py:36] Route: /redoc, Methods: GET, HEAD
INFO 08-05 11:57:34 [launcher.py:36] Route: /health, Methods: GET
INFO 08-05 11:57:34 [launcher.py:36] Route: /load, Methods: GET
INFO 08-05 11:57:34 [launcher.py:36] Route: /ping, Methods: GET, POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /tokenize, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /detokenize, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /v1/models, Methods: GET
INFO 08-05 11:57:34 [launcher.py:36] Route: /version, Methods: GET
INFO 08-05 11:57:34 [launcher.py:36] Route: /v1/chat/completions, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /v1/completions, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /v1/embeddings, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /pooling, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /score, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /v1/score, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /v1/audio/transcriptions, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /rerank, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /v1/rerank, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /v2/rerank, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /invocations, Methods: POST
INFO 08-05 11:57:34 [launcher.py:36] Route: /metrics, Methods: GET
INFO:     Started server process [680514]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

测试是否获取模型成功
# curl http://localhost:5306/v1/models

# 测试一条数据
curl http://localhost:5306/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"checkpoint/merged_dapo_model","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\""}]}'
---

### 🤖 PPO（Proximal Policy Optimization）

```bash
bash run_qwen2-05b_ppo.sh

输出:
+ export CUDA_VISIBLE_DEVICES=1,2
+ CUDA_VISIBLE_DEVICES=1,2
+ export VLLM_USE_V1=1
+ VLLM_USE_V1=1
+ HDFS_ROOT=/workspace/verl/backend/reTool
+ DATA_ROOT=/workspace/verl/backend/reTool
+ dapo_math_17k=/workspace/verl/backend/reTool/dataset/BytedTsinghua/train
+ aime_2024=/workspace/verl/backend/reTool/dataset/Maxwell/validation
+ actor_model_path=./models/merged_sft_model
+ critic_model_path=./models/merged_sft_model
+ train_files='['\''/workspace/verl/backend/reTool/dataset/BytedTsinghua/train'\'']'
+ test_files='['\''/workspace/verl/backend/reTool/dataset/Maxwell/validation'\'']'
+ tool_config_path=./sandbox_fusion_tool_config.yaml
+ project_name=wuxibin_retool
+ experiment_name=qwen2.5-05b_ppo
+ default_local_dir=/workspace/verl/backend/reTool/checkpoint/qwen2.5-05b_ppo
+ adv_estimator=gae
+ use_kl_in_reward=False
+ kl_coef=0.0
+ use_kl_loss=False
+ kl_loss_coef=0.0
+ clip_ratio_low=0.2
+ clip_ratio_high=0.28
+ max_turns=8
+ max_prompt_length=2048
+ max_response_length=4096
+ actor_lr=1e-6
+ critic_lr=2e-6
+ gae_gamma=1.0
+ gae_lam=1.0
+ critic_warmup=20
+ train_batch_size=16
+ ppo_mini_batch_size=8
+ n_resp_per_prompt_val=4
+ infer_tp=1
+ train_sp=2
+ offload=True
+ actor_max_token_len_per_gpu=12288
+ critic_max_token_len_per_gpu=24576
+ python3 -m verl.trainer.main_ppo algorithm.adv_estimator=gae algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 algorithm.gamma=1.0 algorithm.lam=1.0 'data.train_files=['\''/workspace/verl/backend/reTool/dataset/BytedTsinghua/train'\'']' 'data.val_files=['\''/workspace/verl/backend/reTool/dataset/Maxwell/validation'\'']' data.return_raw_chat=True data.train_batch_size=16 data.max_prompt_length=2048 data.max_response_length=4096 data.filter_overlong_prompts=True data.truncation=error data.custom_cls.path=retool.py data.custom_cls.name=CustomRLHFDataset custom_reward_function.path=retool.py custom_reward_function.name=compute_score actor_rollout_ref.model.path=./models/merged_sft_model actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.clip_ratio_c=10.0 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.max_user_turns=8 actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 actor_rollout_ref.rollout.multi_turn.tool_config_path=./sandbox_fusion_tool_config.yaml actor_rollout_ref.rollout.multi_turn.format=hermes actor_rollout_ref.rollout.gpu_memory_utilization=0.9 actor_rollout_ref.rollout.val_kwargs.top_p=0.6 actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.n=4 critic.optim.lr=2e-6 critic.model.use_remove_padding=True critic.model.path=./models/merged_sft_model critic.model.enable_gradient_checkpointing=True critic.ppo_max_token_len_per_gpu=24576 critic.ulysses_sequence_parallel_size=2 critic.model.fsdp_config.param_offload=True critic.model.fsdp_config.optimizer_offload=True trainer.critic_warmup=20 'trainer.logger=[console]' trainer.project_name=wuxibin_retool trainer.experiment_name=qwen2.5-05b_ppo trainer.n_gpus_per_node=2 trainer.val_before_train=True trainer.log_val_generations=100 trainer.nnodes=1 trainer.save_freq=1 trainer.default_local_dir=/workspace/verl/backend/reTool/checkpoint/qwen2.5-05b_ppo trainer.test_freq=5 trainer.total_epochs=1
2025-08-05 12:08:11,644 INFO worker.py:1879 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265
(TaskRunner pid=687114) TaskRunner hostname: yaqiyun-SYS-4028GR-TR2, PID: 687114
(TaskRunner pid=687114) {'actor_rollout_ref': {'actor': {'_target_': 'verl.workers.config.FSDPActorConfig',
(TaskRunner pid=687114)                                  'checkpoint': {'_target_': 'verl.trainer.config.CheckpointConfig',
(TaskRunner pid=687114)                                                 'async_save': False,
(TaskRunner pid=687114)                                                 'load_contents': ['model',
(TaskRunner pid=687114)                                                                   'optimizer',
(TaskRunner pid=687114)                                                                   'extra'],
(TaskRunner pid=687114)                                                 'save_contents': ['model',
(TaskRunner pid=687114)                                                                   'optimizer',
(TaskRunner pid=687114)                                                                   'extra']},
(TaskRunner pid=687114)                                  'clip_ratio': 0.2,
(TaskRunner pid=687114)                                  'clip_ratio_c': 10.0,
(TaskRunner pid=687114)                                  'clip_ratio_high': 0.28,
(TaskRunner pid=687114)                                  'clip_ratio_low': 0.2,
(TaskRunner pid=687114)                                  'entropy_checkpointing': False,
(TaskRunner pid=687114)                                  'entropy_coeff': 0,
(TaskRunner pid=687114)                                  'entropy_from_logits_with_chunking': False,
(TaskRunner pid=687114)                                  'fsdp_config': {'_target_': 'verl.workers.config.FSDPEngineConfig',
(TaskRunner pid=687114)                                                  'forward_prefetch': False,
(TaskRunner pid=687114)                                                  'fsdp_size': -1,
(TaskRunner pid=687114)                                                  'offload_policy': False,
(TaskRunner pid=687114)                                                  'optimizer_offload': True,
(TaskRunner pid=687114)                                                  'param_offload': True,
(TaskRunner pid=687114)                                                  'reshard_after_forward': True,
(TaskRunner pid=687114)                                                  'wrap_policy': {'min_num_params': 0}},
(TaskRunner pid=687114)                                  'grad_clip': 1.0,
(TaskRunner pid=687114)                                  'kl_loss_coef': 0.0,
(TaskRunner pid=687114)                                  'kl_loss_type': 'low_var_kl',
(TaskRunner pid=687114)                                  'loss_agg_mode': 'token-mean',
(TaskRunner pid=687114)                                  'optim': {'_target_': 'verl.workers.config.FSDPOptimizerConfig',
(TaskRunner pid=687114)                                            'lr': 1e-06,
(TaskRunner pid=687114)                                            'lr_warmup_steps': -1,
(TaskRunner pid=687114)                                            'lr_warmup_steps_ratio': 0.0,
(TaskRunner pid=687114)                                            'min_lr_ratio': 0.0,
(TaskRunner pid=687114)                                            'num_cycles': 0.5,
(TaskRunner pid=687114)                                            'total_training_steps': -1,
(TaskRunner pid=687114)                                            'warmup_style': 'constant',
(TaskRunner pid=687114)                                            'weight_decay': 0.01},
(TaskRunner pid=687114)                                  'policy_loss': {'_target_': 'verl.workers.config.PolicyLossConfig',
(TaskRunner pid=687114)                                                  'clip_cov_lb': 1.0,
(TaskRunner pid=687114)                                                  'clip_cov_ratio': 0.0002,
(TaskRunner pid=687114)                                                  'clip_cov_ub': 5.0,
(TaskRunner pid=687114)                                                  'kl_cov_ratio': 0.0002,
(TaskRunner pid=687114)                                                  'loss_mode': 'vanilla',
(TaskRunner pid=687114)                                                  'ppo_kl_coef': 0.1},
(TaskRunner pid=687114)                                  'ppo_epochs': 1,
(TaskRunner pid=687114)                                  'ppo_max_token_len_per_gpu': 12288,
(TaskRunner pid=687114)                                  'ppo_micro_batch_size': None,
(TaskRunner pid=687114)                                  'ppo_micro_batch_size_per_gpu': None,
(TaskRunner pid=687114)                                  'ppo_mini_batch_size': 8,
(TaskRunner pid=687114)                                  'shuffle': False,
(TaskRunner pid=687114)                                  'strategy': 'fsdp',
(TaskRunner pid=687114)                                  'ulysses_sequence_parallel_size': 2,
(TaskRunner pid=687114)                                  'use_dynamic_bsz': True,
(TaskRunner pid=687114)                                  'use_fused_kernels': False,
(TaskRunner pid=687114)                                  'use_kl_loss': False,
(TaskRunner pid=687114)                                  'use_remove_padding': True,
(TaskRunner pid=687114)                                  'use_torch_compile': True},
(TaskRunner pid=687114)                        'hybrid_engine': True,
(TaskRunner pid=687114)                        'model': {'custom_chat_template': None,
(TaskRunner pid=687114)                                  'enable_activation_offload': False,
(TaskRunner pid=687114)                                  'enable_gradient_checkpointing': True,
(TaskRunner pid=687114)                                  'exclude_modules': None,
(TaskRunner pid=687114)                                  'external_lib': None,
(TaskRunner pid=687114)                                  'fused_kernel_options': {'impl_backend': 'torch'},
(TaskRunner pid=687114)                                  'lora_alpha': 16,
(TaskRunner pid=687114)                                  'lora_rank': 0,
(TaskRunner pid=687114)                                  'override_config': {},
(TaskRunner pid=687114)                                  'path': './models/merged_sft_model',
(TaskRunner pid=687114)                                  'target_modules': 'all-linear',
(TaskRunner pid=687114)                                  'trust_remote_code': False,
(TaskRunner pid=687114)                                  'use_fused_kernels': False,
(TaskRunner pid=687114)                                  'use_liger': False,
(TaskRunner pid=687114)                                  'use_remove_padding': True,
(TaskRunner pid=687114)                                  'use_shm': False},
(TaskRunner pid=687114)                        'profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
(TaskRunner pid=687114)                                     'all_ranks': False,
(TaskRunner pid=687114)                                     'discrete': False,
(TaskRunner pid=687114)                                     'ranks': []},
(TaskRunner pid=687114)                        'ref': {'entropy_checkpointing': False,
(TaskRunner pid=687114)                                'entropy_from_logits_with_chunking': False,
(TaskRunner pid=687114)                                'fsdp_config': {'_target_': 'verl.workers.config.FSDPEngineConfig',
(TaskRunner pid=687114)                                                'forward_prefetch': False,
(TaskRunner pid=687114)                                                'param_offload': False,
(TaskRunner pid=687114)                                                'reshard_after_forward': True,
(TaskRunner pid=687114)                                                'wrap_policy': {'min_num_params': 0}},
(TaskRunner pid=687114)                                'log_prob_max_token_len_per_gpu': 12288,
(TaskRunner pid=687114)                                'log_prob_micro_batch_size': None,
(TaskRunner pid=687114)                                'log_prob_micro_batch_size_per_gpu': None,
(TaskRunner pid=687114)                                'log_prob_use_dynamic_bsz': True,
(TaskRunner pid=687114)                                'strategy': 'fsdp',
(TaskRunner pid=687114)                                'ulysses_sequence_parallel_size': 2,
(TaskRunner pid=687114)                                'use_torch_compile': True},
(TaskRunner pid=687114)                        'rollout': {'agent': {'agent_loop_config_path': None,
(TaskRunner pid=687114)                                              'custom_async_server': {'name': None,
(TaskRunner pid=687114)                                                                      'path': None},
(TaskRunner pid=687114)                                              'num_workers': 8},
(TaskRunner pid=687114)                                    'calculate_log_probs': False,
(TaskRunner pid=687114)                                    'disable_log_stats': True,
(TaskRunner pid=687114)                                    'do_sample': True,
(TaskRunner pid=687114)                                    'dtype': 'bfloat16',
(TaskRunner pid=687114)                                    'enable_chunked_prefill': True,
(TaskRunner pid=687114)                                    'enforce_eager': True,
(TaskRunner pid=687114)                                    'engine_kwargs': {'sglang': {'attention_backend': None},
(TaskRunner pid=687114)                                                      'vllm': {'disable_mm_preprocessor_cache': False,
(TaskRunner pid=687114)                                                               'swap_space': None}},
(TaskRunner pid=687114)                                    'free_cache_engine': True,
(TaskRunner pid=687114)                                    'gpu_memory_utilization': 0.9,
(TaskRunner pid=687114)                                    'ignore_eos': False,
(TaskRunner pid=687114)                                    'layered_summon': False,
(TaskRunner pid=687114)                                    'load_format': 'dummy_dtensor',
(TaskRunner pid=687114)                                    'log_prob_max_token_len_per_gpu': 12288,
(TaskRunner pid=687114)                                    'log_prob_micro_batch_size': None,
(TaskRunner pid=687114)                                    'log_prob_micro_batch_size_per_gpu': None,
(TaskRunner pid=687114)                                    'log_prob_use_dynamic_bsz': True,
(TaskRunner pid=687114)                                    'max_model_len': None,
(TaskRunner pid=687114)                                    'max_num_batched_tokens': 8192,
(TaskRunner pid=687114)                                    'max_num_seqs': 1024,
(TaskRunner pid=687114)                                    'mode': 'async',
(TaskRunner pid=687114)                                    'multi_stage_wake_up': False,
(TaskRunner pid=687114)                                    'multi_turn': {'enable': True,
(TaskRunner pid=687114)                                                   'format': 'hermes',
(TaskRunner pid=687114)                                                   'interaction_config_path': None,
(TaskRunner pid=687114)                                                   'max_assistant_turns': 8,
(TaskRunner pid=687114)                                                   'max_parallel_calls': 1,
(TaskRunner pid=687114)                                                   'max_tool_response_length': 256,
(TaskRunner pid=687114)                                                   'max_user_turns': 8,
(TaskRunner pid=687114)                                                   'tokenization_sanity_check_mode': 'strict',
(TaskRunner pid=687114)                                                   'tool_config_path': './sandbox_fusion_tool_config.yaml',
(TaskRunner pid=687114)                                                   'tool_response_truncate_side': 'middle',
(TaskRunner pid=687114)                                                   'use_inference_chat_template': False},
(TaskRunner pid=687114)                                    'n': 1,
(TaskRunner pid=687114)                                    'name': 'vllm',
(TaskRunner pid=687114)                                    'prompt_length': 2048,
(TaskRunner pid=687114)                                    'response_length': 4096,
(TaskRunner pid=687114)                                    'temperature': 1.0,
(TaskRunner pid=687114)                                    'tensor_model_parallel_size': 1,
(TaskRunner pid=687114)                                    'top_k': -1,
(TaskRunner pid=687114)                                    'top_p': 1,
(TaskRunner pid=687114)                                    'trace': {'backend': None,
(TaskRunner pid=687114)                                              'token2text': False},
(TaskRunner pid=687114)                                    'update_weights_bucket_megabytes': 512,
(TaskRunner pid=687114)                                    'val_kwargs': {'do_sample': False,
(TaskRunner pid=687114)                                                   'n': 4,
(TaskRunner pid=687114)                                                   'temperature': 1.0,
(TaskRunner pid=687114)                                                   'top_k': -1,
(TaskRunner pid=687114)                                                   'top_p': 0.6}}},
(TaskRunner pid=687114)  'algorithm': {'_target_': 'verl.trainer.config.AlgoConfig',
(TaskRunner pid=687114)                'adv_estimator': 'gae',
(TaskRunner pid=687114)                'gamma': 1.0,
(TaskRunner pid=687114)                'kl_ctrl': {'_target_': 'verl.trainer.config.KLControlConfig',
(TaskRunner pid=687114)                            'horizon': 10000,
(TaskRunner pid=687114)                            'kl_coef': 0.0,
(TaskRunner pid=687114)                            'target_kl': 0.1,
(TaskRunner pid=687114)                            'type': 'fixed'},
(TaskRunner pid=687114)                'kl_penalty': 'kl',
(TaskRunner pid=687114)                'lam': 1.0,
(TaskRunner pid=687114)                'norm_adv_by_std_in_grpo': True,
(TaskRunner pid=687114)                'pf_ppo': {'reweight_method': 'pow', 'weight_pow': 2.0},
(TaskRunner pid=687114)                'use_kl_in_reward': False,
(TaskRunner pid=687114)                'use_pf_ppo': False},
(TaskRunner pid=687114)  'critic': {'_target_': 'verl.workers.config.FSDPCriticConfig',
(TaskRunner pid=687114)             'checkpoint': {'_target_': 'verl.trainer.config.CheckpointConfig',
(TaskRunner pid=687114)                            'async_save': False,
(TaskRunner pid=687114)                            'load_contents': ['model', 'optimizer', 'extra'],
(TaskRunner pid=687114)                            'save_contents': ['model', 'optimizer', 'extra']},
(TaskRunner pid=687114)             'cliprange_value': 0.5,
(TaskRunner pid=687114)             'enable': None,
(TaskRunner pid=687114)             'forward_max_token_len_per_gpu': 24576,
(TaskRunner pid=687114)             'forward_micro_batch_size': None,
(TaskRunner pid=687114)             'forward_micro_batch_size_per_gpu': None,
(TaskRunner pid=687114)             'grad_clip': 1.0,
(TaskRunner pid=687114)             'loss_agg_mode': 'token-mean',
(TaskRunner pid=687114)             'model': {'_target_': 'verl.workers.config.FSDPCriticModelCfg',
(TaskRunner pid=687114)                       'enable_activation_offload': False,
(TaskRunner pid=687114)                       'enable_gradient_checkpointing': True,
(TaskRunner pid=687114)                       'external_lib': None,
(TaskRunner pid=687114)                       'fsdp_config': {'_target_': 'verl.workers.config.FSDPEngineConfig',
(TaskRunner pid=687114)                                       'forward_prefetch': False,
(TaskRunner pid=687114)                                       'fsdp_size': -1,
(TaskRunner pid=687114)                                       'offload_policy': False,
(TaskRunner pid=687114)                                       'optimizer_offload': True,
(TaskRunner pid=687114)                                       'param_offload': True,
(TaskRunner pid=687114)                                       'reshard_after_forward': True,
(TaskRunner pid=687114)                                       'wrap_policy': {'min_num_params': 0}},
(TaskRunner pid=687114)                       'lora_alpha': 16,
(TaskRunner pid=687114)                       'lora_rank': 0,
(TaskRunner pid=687114)                       'override_config': {},
(TaskRunner pid=687114)                       'path': './models/merged_sft_model',
(TaskRunner pid=687114)                       'target_modules': 'all-linear',
(TaskRunner pid=687114)                       'tokenizer_path': './models/merged_sft_model',
(TaskRunner pid=687114)                       'trust_remote_code': False,
(TaskRunner pid=687114)                       'use_remove_padding': True,
(TaskRunner pid=687114)                       'use_shm': False},
(TaskRunner pid=687114)             'optim': {'_target_': 'verl.workers.config.FSDPOptimizerConfig',
(TaskRunner pid=687114)                       'lr': 2e-06,
(TaskRunner pid=687114)                       'lr_warmup_steps': -1,
(TaskRunner pid=687114)                       'lr_warmup_steps_ratio': 0.0,
(TaskRunner pid=687114)                       'min_lr_ratio': None,
(TaskRunner pid=687114)                       'total_training_steps': -1,
(TaskRunner pid=687114)                       'warmup_style': 'constant',
(TaskRunner pid=687114)                       'weight_decay': 0.01},
(TaskRunner pid=687114)             'ppo_epochs': 1,
(TaskRunner pid=687114)             'ppo_max_token_len_per_gpu': 24576,
(TaskRunner pid=687114)             'ppo_micro_batch_size': None,
(TaskRunner pid=687114)             'ppo_micro_batch_size_per_gpu': None,
(TaskRunner pid=687114)             'ppo_mini_batch_size': 8,
(TaskRunner pid=687114)             'profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
(TaskRunner pid=687114)                          'all_ranks': False,
(TaskRunner pid=687114)                          'discrete': False,
(TaskRunner pid=687114)                          'ranks': []},
(TaskRunner pid=687114)             'rollout_n': 1,
(TaskRunner pid=687114)             'shuffle': False,
(TaskRunner pid=687114)             'strategy': 'fsdp',
(TaskRunner pid=687114)             'ulysses_sequence_parallel_size': 2,
(TaskRunner pid=687114)             'use_dynamic_bsz': True},
(TaskRunner pid=687114)  'custom_reward_function': {'name': 'compute_score', 'path': 'retool.py'},
(TaskRunner pid=687114)  'data': {'custom_cls': {'name': 'CustomRLHFDataset', 'path': 'retool.py'},
(TaskRunner pid=687114)           'datagen': {'name': None, 'path': None},
(TaskRunner pid=687114)           'dataloader_num_workers': 8,
(TaskRunner pid=687114)           'filter_overlong_prompts': True,
(TaskRunner pid=687114)           'filter_overlong_prompts_workers': 1,
(TaskRunner pid=687114)           'image_key': 'images',
(TaskRunner pid=687114)           'max_prompt_length': 2048,
(TaskRunner pid=687114)           'max_response_length': 4096,
(TaskRunner pid=687114)           'prompt_key': 'prompt',
(TaskRunner pid=687114)           'return_full_prompt': False,
(TaskRunner pid=687114)           'return_multi_modal_inputs': True,
(TaskRunner pid=687114)           'return_raw_chat': True,
(TaskRunner pid=687114)           'return_raw_input_ids': False,
(TaskRunner pid=687114)           'reward_fn_key': 'data_source',
(TaskRunner pid=687114)           'sampler': {'class_name': None, 'class_path': None},
(TaskRunner pid=687114)           'shuffle': True,
(TaskRunner pid=687114)           'tokenizer': None,
(TaskRunner pid=687114)           'train_batch_size': 16,
(TaskRunner pid=687114)           'train_files': ['/workspace/verl/backend/reTool/dataset/BytedTsinghua/train'],
(TaskRunner pid=687114)           'truncation': 'error',
(TaskRunner pid=687114)           'trust_remote_code': False,
(TaskRunner pid=687114)           'use_shm': False,
(TaskRunner pid=687114)           'val_batch_size': None,
(TaskRunner pid=687114)           'val_files': ['/workspace/verl/backend/reTool/dataset/Maxwell/validation'],
(TaskRunner pid=687114)           'validation_shuffle': False,
(TaskRunner pid=687114)           'video_key': 'videos'},
(TaskRunner pid=687114)  'ray_init': {'num_cpus': None, 'timeline_json_file': None},
(TaskRunner pid=687114)  'reward_model': {'enable': False,
(TaskRunner pid=687114)                   'forward_max_token_len_per_gpu': 24576,
(TaskRunner pid=687114)                   'launch_reward_fn_async': False,
(TaskRunner pid=687114)                   'max_length': None,
(TaskRunner pid=687114)                   'micro_batch_size': None,
(TaskRunner pid=687114)                   'micro_batch_size_per_gpu': None,
(TaskRunner pid=687114)                   'model': {'external_lib': None,
(TaskRunner pid=687114)                             'fsdp_config': {'_target_': 'verl.workers.config.FSDPEngineConfig',
(TaskRunner pid=687114)                                             'forward_prefetch': False,
(TaskRunner pid=687114)                                             'fsdp_size': -1,
(TaskRunner pid=687114)                                             'param_offload': False,
(TaskRunner pid=687114)                                             'reshard_after_forward': True,
(TaskRunner pid=687114)                                             'wrap_policy': {'min_num_params': 0}},
(TaskRunner pid=687114)                             'input_tokenizer': './models/merged_sft_model',
(TaskRunner pid=687114)                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1',
(TaskRunner pid=687114)                             'trust_remote_code': False,
(TaskRunner pid=687114)                             'use_fused_kernels': False,
(TaskRunner pid=687114)                             'use_remove_padding': False,
(TaskRunner pid=687114)                             'use_shm': False},
(TaskRunner pid=687114)                   'profiler': {'_target_': 'verl.utils.profiler.ProfilerConfig',
(TaskRunner pid=687114)                                'all_ranks': False,
(TaskRunner pid=687114)                                'discrete': False,
(TaskRunner pid=687114)                                'ranks': []},
(TaskRunner pid=687114)                   'reward_manager': 'naive',
(TaskRunner pid=687114)                   'sandbox_fusion': {'max_concurrent': 64,
(TaskRunner pid=687114)                                      'memory_limit_mb': 1024,
(TaskRunner pid=687114)                                      'url': None},
(TaskRunner pid=687114)                   'strategy': 'fsdp',
(TaskRunner pid=687114)                   'ulysses_sequence_parallel_size': 1,
(TaskRunner pid=687114)                   'use_dynamic_bsz': True},
(TaskRunner pid=687114)  'trainer': {'balance_batch': True,
(TaskRunner pid=687114)              'controller_nsight_options': {'cuda-graph-trace': 'graph',
(TaskRunner pid=687114)                                            'cuda-memory-usage': 'true',
(TaskRunner pid=687114)                                            'trace': 'cuda,nvtx,cublas,ucx'},
(TaskRunner pid=687114)              'critic_warmup': 20,
(TaskRunner pid=687114)              'default_hdfs_dir': None,
(TaskRunner pid=687114)              'default_local_dir': '/workspace/verl/backend/reTool/checkpoint/qwen2.5-05b_ppo',
(TaskRunner pid=687114)              'del_local_ckpt_after_load': False,
(TaskRunner pid=687114)              'device': 'cuda',
(TaskRunner pid=687114)              'esi_redundant_time': 0,
(TaskRunner pid=687114)              'experiment_name': 'qwen2.5-05b_ppo',
(TaskRunner pid=687114)              'log_val_generations': 100,
(TaskRunner pid=687114)              'logger': ['console'],
(TaskRunner pid=687114)              'max_actor_ckpt_to_keep': None,
(TaskRunner pid=687114)              'max_critic_ckpt_to_keep': None,
(TaskRunner pid=687114)              'n_gpus_per_node': 2,
(TaskRunner pid=687114)              'nnodes': 1,
(TaskRunner pid=687114)              'npu_profile': {'options': {'analysis': True,
(TaskRunner pid=687114)                                          'level': 'level1',
(TaskRunner pid=687114)                                          'record_shapes': False,
(TaskRunner pid=687114)                                          'roles': ['all'],
(TaskRunner pid=687114)                                          'save_path': './profiler_data',
(TaskRunner pid=687114)                                          'with_cpu': True,
(TaskRunner pid=687114)                                          'with_memory': False,
(TaskRunner pid=687114)                                          'with_module': False,
(TaskRunner pid=687114)                                          'with_npu': True,
(TaskRunner pid=687114)                                          'with_stack': False}},
(TaskRunner pid=687114)              'profile_continuous_steps': False,
(TaskRunner pid=687114)              'profile_steps': None,
(TaskRunner pid=687114)              'project_name': 'wuxibin_retool',
(TaskRunner pid=687114)              'ray_wait_register_center_timeout': 300,
(TaskRunner pid=687114)              'resume_from_path': None,
(TaskRunner pid=687114)              'resume_mode': 'auto',
(TaskRunner pid=687114)              'rollout_data_dir': None,
(TaskRunner pid=687114)              'save_freq': 1,
(TaskRunner pid=687114)              'test_freq': 5,
(TaskRunner pid=687114)              'total_epochs': 1,
(TaskRunner pid=687114)              'total_training_steps': None,
(TaskRunner pid=687114)              'use_legacy_worker_impl': 'auto',
(TaskRunner pid=687114)              'val_before_train': True,
(TaskRunner pid=687114)              'val_only': False,
(TaskRunner pid=687114)              'validation_data_dir': None,
(TaskRunner pid=687114)              'worker_nsight_options': {'capture-range': 'cudaProfilerApi',
(TaskRunner pid=687114)                                        'capture-range-end': None,
(TaskRunner pid=687114)                                        'cuda-graph-trace': 'graph',
(TaskRunner pid=687114)                                        'cuda-memory-usage': 'true',
(TaskRunner pid=687114)                                        'kill': 'none',
(TaskRunner pid=687114)                                        'trace': 'cuda,nvtx,cublas,ucx'}}}
(TaskRunner pid=687114) using customized reward function 'compute_score' from 'retool.py'
(TaskRunner pid=687114) using customized reward function 'compute_score' from 'retool.py'
(TaskRunner pid=687114) Using dataset class: CustomRLHFDataset
(TaskRunner pid=687114) 加载parquet_file: /workspace/verl/backend/reTool/dataset/BytedTsinghua/train
(TaskRunner pid=687114) Setting TOKENIZERS_PARALLELISM=false for forked processes.
(TaskRunner pid=687114) WARNING:2025-08-05 12:08:26,588:Setting TOKENIZERS_PARALLELISM=false for forked processes.
Map (num_proc=16):   0%|          | 0/1791700 [00:00<?, ? examples/s]
Map (num_proc=16):   0%|          | 385/1791700 [00:01<1:21:52, 364.62 examples/s]
Map (num_proc=16):   0%|          | 1430/1791700 [00:01<19:22, 1540.23 examples/s]
Map (num_proc=16):   0%|          | 2000/1791700 [00:01<15:25, 1932.80 examples/s]
Map (num_proc=16):   0%|          | 4084/1791700 [00:01<06:26, 4619.47 examples/s]
Map (num_proc=16):   0%|          | 5191/1791700 [00:01<05:37, 5286.20 examples/s]
Map (num_proc=16):   0%|          | 8095/1791700 [00:01<03:06, 9538.52 examples/s]
Map (num_proc=16):   1%|          | 9827/1791700 [00:01<03:05, 9607.27 examples/s]
Map (num_proc=16):   1%|          | 14637/1791700 [00:02<01:42, 17282.07 examples/s]
Map (num_proc=16):   1%|          | 17423/1791700 [00:02<01:45, 16780.52 examples/s]
Map (num_proc=16):   1%|▏         | 22680/1791700 [00:02<01:13, 24207.35 examples/s]
Map (num_proc=16):   1%|▏         | 25734/1791700 [00:02<01:23, 21151.65 examples/s]
Map (num_proc=16):   2%|▏         | 33579/1791700 [00:02<00:53, 32880.57 examples/s]
Map (num_proc=16):   2%|▏         | 37911/1791700 [00:02<01:00, 29092.59 examples/s]
Map (num_proc=16):   3%|▎         | 45609/1791700 [00:02<00:45, 38640.32 examples/s]
Map (num_proc=16):   3%|▎         | 50177/1791700 [00:03<00:51, 34125.61 examples/s]
Map (num_proc=16):   3%|▎         | 60314/1791700 [00:03<00:36, 47881.22 examples/s]
Map (num_proc=16):   4%|▎         | 66132/1791700 [00:03<00:40, 42189.60 examples/s]
Map (num_proc=16):   4%|▍         | 75812/1791700 [00:03<00:32, 53525.32 examples/s]
Map (num_proc=16):   5%|▍         | 82347/1791700 [00:03<00:35, 48054.64 examples/s]
Map (num_proc=16):   5%|▌         | 94280/1791700 [00:03<00:27, 62106.57 examples/s]
Map (num_proc=16):   6%|▌         | 101563/1791700 [00:03<00:29, 56884.44 examples/s]
Map (num_proc=16):   6%|▋         | 114502/1791700 [00:04<00:22, 73276.80 examples/s]
Map (num_proc=16):   7%|▋         | 123037/1791700 [00:04<00:25, 65681.16 examples/s]
Map (num_proc=16):   8%|▊         | 134885/1791700 [00:04<00:21, 77888.93 examples/s]
Map (num_proc=16):   8%|▊         | 143854/1791700 [00:04<00:23, 69103.95 examples/s]
Map (num_proc=16):   9%|▉         | 161677/1791700 [00:04<00:17, 92575.79 examples/s]
Map (num_proc=16):  10%|▉         | 172044/1791700 [00:04<00:17, 90392.13 examples/s]
Map (num_proc=16):  10%|█         | 181966/1791700 [00:04<00:18, 88249.97 examples/s]
Map (num_proc=16):  11%|█         | 191629/1791700 [00:04<00:18, 84282.84 examples/s]
Map (num_proc=16):  11%|█▏        | 202547/1791700 [00:05<00:17, 89890.07 examples/s]
Map (num_proc=16):  12%|█▏        | 212230/1791700 [00:05<00:17, 91079.24 examples/s]
Map (num_proc=16):  12%|█▏        | 221989/1791700 [00:05<00:17, 89564.61 examples/s]
Map (num_proc=16):  13%|█▎        | 233068/1791700 [00:05<00:16, 94833.33 examples/s]
Map (num_proc=16):  14%|█▎        | 243621/1791700 [00:05<00:15, 97763.71 examples/s]
Map (num_proc=16):  14%|█▍        | 254141/1791700 [00:05<00:15, 99865.23 examples/s]
Map (num_proc=16):  15%|█▍        | 265236/1791700 [00:05<00:14, 102053.02 examples/s]
Map (num_proc=16):  15%|█▌        | 277111/1791700 [00:05<00:14, 106480.27 examples/s]
Map (num_proc=16):  16%|█▌        | 288889/1791700 [00:05<00:14, 106705.00 examples/s]
Map (num_proc=16):  17%|█▋        | 299806/1791700 [00:05<00:14, 106287.49 examples/s]
Map (num_proc=16):  17%|█▋        | 311004/1791700 [00:06<00:13, 106935.05 examples/s]
Map (num_proc=16):  18%|█▊        | 322711/1791700 [00:06<00:13, 109321.16 examples/s]
Map (num_proc=16):  19%|█▊        | 333904/1791700 [00:06<00:13, 108038.11 examples/s]
Map (num_proc=16):  19%|█▉        | 345097/1791700 [00:06<00:13, 104280.63 examples/s]
Map (num_proc=16):  20%|█▉        | 356154/1791700 [00:06<00:13, 104326.35 examples/s]
Map (num_proc=16):  20%|██        | 366895/1791700 [00:06<00:13, 101977.72 examples/s]
Map (num_proc=16):  21%|██        | 378374/1791700 [00:06<00:13, 105278.39 examples/s]
Map (num_proc=16):  22%|██▏       | 389260/1791700 [00:06<00:13, 102406.50 examples/s]
Map (num_proc=16):  22%|██▏       | 400561/1791700 [00:06<00:13, 104589.44 examples/s]
Map (num_proc=16):  23%|██▎       | 411283/1791700 [00:07<00:13, 101365.60 examples/s]
Map (num_proc=16):  24%|██▍       | 433166/1791700 [00:07<00:12, 104710.53 examples/s]
Map (num_proc=16):  25%|██▍       | 443935/1791700 [00:07<00:13, 102269.27 examples/s]
Map (num_proc=16):  25%|██▌       | 454628/1791700 [00:07<00:13, 100437.32 examples/s]
Map (num_proc=16):  26%|██▌       | 465254/1791700 [00:07<00:12, 102071.91 examples/s]
Map (num_proc=16):  27%|██▋       | 477021/1791700 [00:07<00:12, 105386.30 examples/s]
Map (num_proc=16):  27%|██▋       | 487872/1791700 [00:07<00:12, 104307.53 examples/s]
Map (num_proc=16):  28%|██▊       | 498479/1791700 [00:07<00:12, 103282.56 examples/s]
Map (num_proc=16):  28%|██▊       | 508876/1791700 [00:08<00:12, 99946.31 examples/s]
Map (num_proc=16):  29%|██▉       | 519212/1791700 [00:08<00:12, 100409.79 examples/s]
Map (num_proc=16):  30%|██▉       | 529725/1791700 [00:08<00:12, 101112.63 examples/s]
Map (num_proc=16):  30%|███       | 541621/1791700 [00:08<00:11, 106150.33 examples/s]
Map (num_proc=16):  31%|███       | 552861/1791700 [00:08<00:11, 107961.94 examples/s]
Map (num_proc=16):  31%|███▏      | 564241/1791700 [00:08<00:11, 108321.89 examples/s]
Map (num_proc=16):  32%|███▏      | 577249/1791700 [00:08<00:10, 113542.03 examples/s]
Map (num_proc=16):  33%|███▎      | 588810/1791700 [00:08<00:10, 110191.31 examples/s]
Map (num_proc=16):  33%|███▎      | 600193/1791700 [00:08<00:10, 110003.60 examples/s]
Map (num_proc=16):  34%|███▍      | 611406/1791700 [00:08<00:11, 102648.09 examples/s]
Map (num_proc=16):  35%|███▍      | 621996/1791700 [00:09<00:11, 101484.88 examples/s]
Map (num_proc=16):  35%|███▌      | 634453/1791700 [00:09<00:10, 107014.56 examples/s]
Map (num_proc=16):  36%|███▌      | 645424/1791700 [00:09<00:10, 105513.15 examples/s]
Map (num_proc=16):  37%|███▋      | 656412/1791700 [00:09<00:11, 101489.99 examples/s]
Map (num_proc=16):  37%|███▋      | 667804/1791700 [00:09<00:10, 104660.22 examples/s]
Map (num_proc=16):  38%|███▊      | 678459/1791700 [00:09<00:10, 103420.31 examples/s]
Map (num_proc=16):  39%|███▊      | 690519/1791700 [00:09<00:10, 108207.42 examples/s]
Map (num_proc=16):  39%|███▉      | 701698/1791700 [00:09<00:10, 108278.12 examples/s]
Map (num_proc=16):  40%|███▉      | 712873/1791700 [00:09<00:10, 105769.50 examples/s]
Map (num_proc=16):  40%|████      | 724433/1791700 [00:10<00:09, 108535.81 examples/s]
Map (num_proc=16):  41%|████      | 735943/1791700 [00:10<00:09, 109589.75 examples/s]
Map (num_proc=16):  42%|████▏     | 747299/1791700 [00:10<00:09, 108260.30 examples/s]
Map (num_proc=16):  42%|████▏     | 758482/1791700 [00:10<00:09, 106225.22 examples/s]
Map (num_proc=16):  43%|████▎     | 770551/1791700 [00:10<00:09, 110331.25 examples/s]
Map (num_proc=16):  44%|████▎     | 781740/1791700 [00:10<00:09, 110548.38 examples/s]
Map (num_proc=16):  44%|████▍     | 792897/1791700 [00:10<00:09, 103567.04 examples/s]
Map (num_proc=16):  46%|████▌     | 815773/1791700 [00:10<00:09, 105924.48 examples/s]
Map (num_proc=16):  46%|████▌     | 826793/1791700 [00:10<00:09, 104376.01 examples/s]
Map (num_proc=16):  47%|████▋     | 837467/1791700 [00:11<00:09, 104180.12 examples/s]
Map (num_proc=16):  47%|████▋     | 848008/1791700 [00:11<00:09, 101963.04 examples/s]
Map (num_proc=16):  48%|████▊     | 859667/1791700 [00:11<00:08, 105945.12 examples/s]
Map (num_proc=16):  49%|████▊     | 870447/1791700 [00:11<00:08, 105165.81 examples/s]
Map (num_proc=16):  49%|████▉     | 881418/1791700 [00:11<00:08, 105989.92 examples/s]
Map (num_proc=16):  50%|████▉     | 892493/1791700 [00:11<00:08, 107286.43 examples/s]
Map (num_proc=16):  50%|█████     | 903662/1791700 [00:11<00:08, 105093.03 examples/s]
Map (num_proc=16):  51%|█████     | 914343/1791700 [00:11<00:08, 102124.65 examples/s]
Map (num_proc=16):  52%|█████▏    | 925058/1791700 [00:11<00:08, 102829.70 examples/s]
Map (num_proc=16):  52%|█████▏    | 935669/1791700 [00:12<00:08, 97084.69 examples/s]
Map (num_proc=16):  53%|█████▎    | 945625/1791700 [00:12<00:08, 97236.53 examples/s]
Map (num_proc=16):  53%|█████▎    | 956104/1791700 [00:12<00:08, 99371.62 examples/s]
Map (num_proc=16):  54%|█████▍    | 967852/1791700 [00:12<00:07, 104505.24 examples/s]
Map (num_proc=16):  55%|█████▍    | 979053/1791700 [00:12<00:07, 106676.30 examples/s]
Map (num_proc=16):  55%|█████▌    | 989957/1791700 [00:12<00:07, 107334.00 examples/s]
Map (num_proc=16):  56%|█████▌    | 1000808/1791700 [00:12<00:07, 107402.50 examples/s]
Map (num_proc=16):  57%|█████▋    | 1012426/1791700 [00:12<00:07, 109960.68 examples/s]
Map (num_proc=16):  57%|█████▋    | 1023641/1791700 [00:12<00:07, 106996.37 examples/s]
Map (num_proc=16):  58%|█████▊    | 1034507/1791700 [00:12<00:07, 102469.16 examples/s]
Map (num_proc=16):  58%|█████▊    | 1044942/1791700 [00:13<00:07, 102641.95 examples/s]
Map (num_proc=16):  59%|█████▉    | 1056393/1791700 [00:13<00:06, 105829.26 examples/s]
Map (num_proc=16):  60%|█████▉    | 1068486/1791700 [00:13<00:06, 110171.72 examples/s]
Map (num_proc=16):  60%|██████    | 1079584/1791700 [00:13<00:06, 107261.52 examples/s]
Map (num_proc=16):  61%|██████    | 1090413/1791700 [00:13<00:07, 98422.07 examples/s]
Map (num_proc=16):  61%|██████▏   | 1100691/1791700 [00:13<00:07, 96395.64 examples/s]
Map (num_proc=16):  62%|██████▏   | 1111494/1791700 [00:13<00:06, 98982.19 examples/s]
Map (num_proc=16):  63%|██████▎   | 1123323/1791700 [00:13<00:06, 102437.36 examples/s]
Map (num_proc=16):  63%|██████▎   | 1134925/1791700 [00:13<00:06, 106227.38 examples/s]
Map (num_proc=16):  64%|██████▍   | 1145643/1791700 [00:14<00:06, 104828.32 examples/s]
Map (num_proc=16):  65%|██████▍   | 1157673/1791700 [00:14<00:05, 109198.85 examples/s]
Map (num_proc=16):  65%|██████▌   | 1169076/1791700 [00:14<00:05, 110268.85 examples/s]
Map (num_proc=16):  66%|██████▌   | 1181370/1791700 [00:14<00:05, 113965.94 examples/s]
Map (num_proc=16):  67%|██████▋   | 1192847/1791700 [00:14<00:05, 107797.95 examples/s]
Map (num_proc=16):  67%|██████▋   | 1203848/1791700 [00:14<00:05, 104695.04 examples/s]
Map (num_proc=16):  68%|██████▊   | 1214725/1791700 [00:14<00:05, 105785.70 examples/s]
Map (num_proc=16):  68%|██████▊   | 1226197/1791700 [00:14<00:05, 108323.86 examples/s]
Map (num_proc=16):  69%|██████▉   | 1237410/1791700 [00:14<00:05, 106204.13 examples/s]
Map (num_proc=16):  70%|██████▉   | 1249113/1791700 [00:15<00:04, 109240.29 examples/s]
Map (num_proc=16):  70%|███████   | 1260102/1791700 [00:15<00:05, 104526.86 examples/s]
Map (num_proc=16):  71%|███████   | 1271024/1791700 [00:15<00:04, 105781.44 examples/s]
Map (num_proc=16):  72%|███████▏  | 1281913/1791700 [00:15<00:04, 102240.52 examples/s]
Map (num_proc=16):  72%|███████▏  | 1292604/1791700 [00:15<00:04, 102370.26 examples/s]
Map (num_proc=16):  73%|███████▎  | 1304477/1791700 [00:15<00:04, 105433.31 examples/s]
Map (num_proc=16):  73%|███████▎  | 1316393/1791700 [00:15<00:04, 109360.24 examples/s]
Map (num_proc=16):  74%|███████▍  | 1328376/1791700 [00:15<00:04, 112307.93 examples/s]
Map (num_proc=16):  75%|███████▍  | 1339832/1791700 [00:15<00:04, 103116.63 examples/s]
Map (num_proc=16):  75%|███████▌  | 1350470/1791700 [00:16<00:04, 91442.82 examples/s]
Map (num_proc=16):  76%|███████▌  | 1360069/1791700 [00:16<00:04, 92603.34 examples/s]
Map (num_proc=16):  77%|███████▋  | 1371447/1791700 [00:16<00:04, 98170.60 examples/s]
Map (num_proc=16):  77%|███████▋  | 1382937/1791700 [00:16<00:04, 101671.17 examples/s]
Map (num_proc=16):  78%|███████▊  | 1393919/1791700 [00:16<00:03, 103898.70 examples/s]
Map (num_proc=16):  78%|███████▊  | 1404984/1791700 [00:16<00:03, 105810.63 examples/s]
Map (num_proc=16):  79%|███████▉  | 1415705/1791700 [00:16<00:03, 105918.76 examples/s]
Map (num_proc=16):  80%|███████▉  | 1426641/1791700 [00:16<00:03, 106655.78 examples/s]
Map (num_proc=16):  80%|████████  | 1437874/1791700 [00:16<00:03, 108298.49 examples/s]
Map (num_proc=16):  81%|████████  | 1448991/1791700 [00:16<00:03, 106356.48 examples/s]
Map (num_proc=16):  81%|████████▏ | 1459753/1791700 [00:17<00:03, 106689.41 examples/s]
Map (num_proc=16):  82%|████████▏ | 1470729/1791700 [00:17<00:03, 106243.68 examples/s]
Map (num_proc=16):  83%|████████▎ | 1481859/1791700 [00:17<00:02, 107692.80 examples/s]
Map (num_proc=16):  83%|████████▎ | 1492975/1791700 [00:17<00:02, 103179.61 examples/s]
Map (num_proc=16):  84%|████████▍ | 1503370/1791700 [00:17<00:02, 102966.20 examples/s]
Map (num_proc=16):  84%|████████▍ | 1513877/1791700 [00:17<00:02, 103538.90 examples/s]
Map (num_proc=16):  85%|████████▌ | 1524493/1791700 [00:17<00:02, 100325.11 examples/s]
Map (num_proc=16):  86%|████████▌ | 1536123/1791700 [00:17<00:02, 104301.26 examples/s]
Map (num_proc=16):  86%|████████▋ | 1547093/1791700 [00:17<00:02, 105836.37 examples/s]
Map (num_proc=16):  87%|████████▋ | 1558094/1791700 [00:17<00:02, 104679.03 examples/s]
Map (num_proc=16):  88%|████████▊ | 1568697/1791700 [00:18<00:02, 103795.47 examples/s]
Map (num_proc=16):  88%|████████▊ | 1579349/1791700 [00:18<00:02, 95609.84 examples/s]
Map (num_proc=16):  89%|████████▊ | 1589355/1791700 [00:18<00:02, 86916.92 examples/s]
Map (num_proc=16):  89%|████████▉ | 1599365/1791700 [00:18<00:02, 89926.13 examples/s]
Map (num_proc=16):  90%|████████▉ | 1608930/1791700 [00:18<00:02, 80912.06 examples/s]
Map (num_proc=16):  90%|█████████ | 1617361/1791700 [00:18<00:02, 73622.54 examples/s]
Map (num_proc=16):  91%|█████████ | 1625207/1791700 [00:18<00:02, 74041.55 examples/s]
Map (num_proc=16):  91%|█████████ | 1633822/1791700 [00:18<00:02, 76903.65 examples/s]
Map (num_proc=16):  92%|█████████▏| 1641828/1791700 [00:19<00:02, 69590.28 examples/s]
Map (num_proc=16):  92%|█████████▏| 1649066/1791700 [00:19<00:02, 69065.92 examples/s]
Map (num_proc=16):  92%|█████████▏| 1656374/1791700 [00:19<00:02, 67617.33 examples/s]
Map (num_proc=16):  93%|█████████▎| 1663234/1791700 [00:19<00:02, 60075.44 examples/s]
Map (num_proc=16):  93%|█████████▎| 1670661/1791700 [00:19<00:01, 63412.88 examples/s]
Map (num_proc=16):  94%|█████████▎| 1677303/1791700 [00:19<00:01, 61376.84 examples/s]
Map (num_proc=16):  94%|█████████▍| 1685603/1791700 [00:19<00:01, 64689.13 examples/s]
Map (num_proc=16):  94%|█████████▍| 1692197/1791700 [00:19<00:01, 61650.40 examples/s]
Map (num_proc=16):  95%|█████████▍| 1698442/1791700 [00:20<00:01, 55497.87 examples/s]
Map (num_proc=16):  95%|█████████▌| 1704225/1791700 [00:20<00:01, 54453.96 examples/s]
Map (num_proc=16):  95%|█████████▌| 1709863/1791700 [00:20<00:01, 48000.74 examples/s]
Map (num_proc=16):  96%|█████████▌| 1715033/1791700 [00:20<00:01, 44714.32 examples/s]
Map (num_proc=16):  96%|█████████▌| 1719909/1791700 [00:20<00:01, 41167.09 examples/s]
Map (num_proc=16):  96%|█████████▌| 1724210/1791700 [00:20<00:01, 41250.96 examples/s]
Map (num_proc=16):  96%|█████████▋| 1728517/1791700 [00:20<00:01, 40725.29 examples/s]
Map (num_proc=16):  97%|█████████▋| 1733001/1791700 [00:20<00:01, 40668.38 examples/s]
Map (num_proc=16):  97%|█████████▋| 1737515/1791700 [00:21<00:01, 38425.09 examples/s]
Map (num_proc=16):  97%|█████████▋| 1741919/1791700 [00:21<00:01, 37835.27 examples/s]
Map (num_proc=16):  97%|█████████▋| 1746567/1791700 [00:21<00:01, 39697.39 examples/s]
Map (num_proc=16):  98%|█████████▊| 1750659/1791700 [00:21<00:01, 36911.99 examples/s]
Map (num_proc=16):  98%|█████████▊| 1754827/1791700 [00:21<00:01, 34793.80 examples/s]
Map (num_proc=16):  98%|█████████▊| 1758670/1791700 [00:21<00:00, 33629.87 examples/s]
Map (num_proc=16):  98%|█████████▊| 1762445/1791700 [00:21<00:00, 31394.81 examples/s]
Map (num_proc=16):  99%|█████████▊| 1765632/1791700 [00:21<00:00, 31212.78 examples/s]
Map (num_proc=16):  99%|█████████▊| 1769124/1791700 [00:22<00:00, 29794.74 examples/s]
Map (num_proc=16):  99%|█████████▉| 1772235/1791700 [00:22<00:00, 24524.21 examples/s]
Map (num_proc=16):  99%|█████████▉| 1774872/1791700 [00:22<00:00, 22803.44 examples/s]
Map (num_proc=16):  99%|█████████▉| 1777382/1791700 [00:22<00:00, 16690.11 examples/s]
Map (num_proc=16):  99%|█████████▉| 1779486/1791700 [00:22<00:00, 13346.92 examples/s]
Map (num_proc=16):  99%|█████████▉| 1781222/1791700 [00:23<00:00, 11522.95 examples/s]
Map (num_proc=16):  99%|█████████▉| 1782593/1791700 [00:23<00:00, 10812.92 examples/s]
Map (num_proc=16): 100%|█████████▉| 1784093/1791700 [00:23<00:00, 11523.45 examples/s]
Map (num_proc=16): 100%|█████████▉| 1785674/1791700 [00:23<00:00, 11285.02 examples/s]
Map (num_proc=16): 100%|█████████▉| 1786900/1791700 [00:23<00:00, 11486.42 examples/s]
Map (num_proc=16): 100%|█████████▉| 1790133/1791700 [00:23<00:00, 13370.06 examples/s]
Map (num_proc=16): 100%|██████████| 1791700/1791700 [00:24<00:00, 5357.12 examples/s]
Map (num_proc=16): 100%|██████████| 1791700/1791700 [00:24<00:00, 72449.30 examples/s]
(TaskRunner pid=687114) dataset len: 1791700
(TaskRunner pid=687114) Using dataset class: CustomRLHFDataset
(TaskRunner pid=687114) 加载parquet_file: /workspace/verl/backend/reTool/dataset/Maxwell/validation
(TaskRunner pid=687114) Setting TOKENIZERS_PARALLELISM=false for forked processes.
(TaskRunner pid=687114) WARNING:2025-08-05 12:08:52,124:Setting TOKENIZERS_PARALLELISM=false for forked processes.
Map (num_proc=16):   0%|          | 0/30 [00:00<?, ? examples/s]
Map (num_proc=16):   7%|▋         | 2/30 [00:00<00:12,  2.23 examples/s]
Map (num_proc=16):  13%|█▎        | 4/30 [00:01<00:07,  3.64 examples/s]
Map (num_proc=16):  20%|██        | 6/30 [00:01<00:05,  4.25 examples/s]
Map (num_proc=16):  33%|███▎      | 10/30 [00:01<00:02,  6.92 examples/s]
Map (num_proc=16):  40%|████      | 12/30 [00:02<00:02,  7.37 examples/s]
Map (num_proc=16):  47%|████▋     | 14/30 [00:02<00:02,  6.71 examples/s]
Map (num_proc=16):  53%|█████▎    | 16/30 [00:02<00:01,  7.16 examples/s]
Map (num_proc=16):  60%|██████    | 18/30 [00:02<00:01,  7.64 examples/s]
Map (num_proc=16):  67%|██████▋   | 20/30 [00:03<00:01,  7.98 examples/s]
Map (num_proc=16):  73%|███████▎  | 22/30 [00:03<00:00,  8.12 examples/s]
Map (num_proc=16):  80%|████████  | 24/30 [00:03<00:00,  8.19 examples/s]
Map (num_proc=16):  93%|█████████▎| 28/30 [00:03<00:00, 10.24 examples/s]
Map (num_proc=16): 100%|██████████| 30/30 [00:04<00:00,  7.78 examples/s]
Map (num_proc=16): 100%|██████████| 30/30 [00:04<00:00,  6.74 examples/s]
(TaskRunner pid=687114) dataset len: 30
(TaskRunner pid=687114) minimal_bsz 2
(TaskRunner pid=687114) real_train_batch_size 16
(TaskRunner pid=687114) DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
(TaskRunner pid=687114) [validate_config] All configuration checks passed successfully!
(TaskRunner pid=687114) Size of train dataloader: 111981, Size of val dataloader: 1
(TaskRunner pid=687114) Total training steps: 111981
(TaskRunner pid=687114) colocated worker base class <class 'verl.single_controller.base.worker.Worker'>
(TaskRunner pid=687114) bind role actor_rollout method chat_completion to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(TaskRunner pid=687114) bind role actor_rollout method execute_method to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(TaskRunner pid=687114) bind role actor_rollout method generate to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(TaskRunner pid=687114) bind role actor_rollout method get_zeromq_address to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(TaskRunner pid=687114) bind role actor_rollout method sleep to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(TaskRunner pid=687114) bind role actor_rollout method wake_up to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(TaskRunner pid=687114) WARNING:2025-08-05 12:08:59,122:Waiting for register center actor 9aMJ77_register_center to be ready. Elapsed time: 0 seconds out of 300 seconds.
(pid=689549) Using blocking ray.get inside async actor. This blocks the event loop. Please use `await` on object ref with asyncio.gather if you want to yield execution to the event loop instead.
(pid=689739) Using blocking ray.get inside async actor. This blocks the event loop. Please use `await` on object ref with asyncio.gather if you want to yield execution to the event loop instead.
(WorkerDict pid=689549) device_name cuda
(WorkerDict pid=689549) dp 1
(WorkerDict pid=689549) self.ulysses_sequence_parallel_size 2
(WorkerDict pid=689549) Critic overriding config {'bos_token_id': None, 'eos_token_id': 151645, 'pad_token_id': 151643}
(WorkerDict pid=689549) Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForTokenClassification is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
(WorkerDict pid=689549) You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
(WorkerDict pid=689549) Some weights of Qwen2ForTokenClassification were not initialized from the model checkpoint at ./models/merged_sft_model and are newly initialized: ['score.bias', 'score.weight']
(WorkerDict pid=689549) You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
(WorkerDict pid=689549) Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
(WorkerDict pid=689549) Skipping monkey patch for Qwen2ForTokenClassification as use_fused_kernels is False or fused_kernels_backend is None
(WorkerDict pid=689549) Qwen2ForTokenClassification contains 494.03M parameters
(WorkerDict pid=689549) Before critic FSDP, memory allocated (GB): 0.00, memory reserved (GB): 0.00, device memory used/total (GB): 0.36/23.55
(WorkerDict pid=689549) NCCL version 2.21.5+cuda12.4
(WorkerDict pid=689549) After critic FSDP, memory allocated (GB): 0.92, memory reserved (GB): 2.67, device memory used/total (GB): 3.21/23.55
(WorkerDict pid=689549) Total steps: 111981, num_warmup_steps: 0
(WorkerDict pid=689739) device_name cuda
(WorkerDict pid=689739) dp 1
(WorkerDict pid=689739) self.ulysses_sequence_parallel_size 2
(WorkerDict pid=689549) Critic use_remove_padding=True
(WorkerDict pid=689549) Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)` [repeated 2x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(WorkerDict pid=689739) You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
(WorkerDict pid=689549) Model config after override: Qwen2Config {
(WorkerDict pid=689549)   "architectures": [
(WorkerDict pid=689549)     "Qwen2ForCausalLM"
(WorkerDict pid=689549)   ],
(WorkerDict pid=689549)   "attention_dropout": 0.0,
(WorkerDict pid=689549)   "eos_token_id": 151645,
(WorkerDict pid=689549)   "hidden_act": "silu",
(WorkerDict pid=689549)   "hidden_size": 896,
(WorkerDict pid=689549)   "initializer_range": 0.02,
(WorkerDict pid=689549)   "intermediate_size": 4864,
(WorkerDict pid=689549)   "max_position_embeddings": 32768,
(WorkerDict pid=689549)   "max_window_layers": 21,
(WorkerDict pid=689549)   "model_type": "qwen2",
(WorkerDict pid=689549)   "num_attention_heads": 14,
(WorkerDict pid=689549)   "num_hidden_layers": 24,
(WorkerDict pid=689549)   "num_key_value_heads": 2,
(WorkerDict pid=689549)   "pad_token_id": 151643,
(WorkerDict pid=689549)   "rms_norm_eps": 1e-06,
(WorkerDict pid=689549)   "rope_scaling": null,
(WorkerDict pid=689549)   "rope_theta": 1000000.0,
(WorkerDict pid=689549)   "sliding_window": 32768,
(WorkerDict pid=689549)   "tie_word_embeddings": true,
(WorkerDict pid=689549)   "torch_dtype": "bfloat16",
(WorkerDict pid=689549)   "transformers_version": "4.51.1",
(WorkerDict pid=689549)   "use_cache": true,
(WorkerDict pid=689549)   "use_sliding_window": false,
(WorkerDict pid=689549)   "vocab_size": 151936
(WorkerDict pid=689549) }
(WorkerDict pid=689549)
(WorkerDict pid=689549) actor_module de local_path ./models/merged_sft_model
(WorkerDict pid=689549) Skipping monkey patch for Qwen2ForCausalLM as use_fused_kernels is False or fused_kernels_backend is torch
(WorkerDict pid=689549) Monkey patch _flash_attention_forward in transformers.integrations.flash_attention [repeated 2x across cluster]
(WorkerDict pid=689739) Skipping monkey patch for Qwen2ForTokenClassification as use_fused_kernels is False or fused_kernels_backend is None
(WorkerDict pid=689549) Qwen2ForCausalLM contains 494.03M parameters
(WorkerDict pid=689549) wrap_policy: functools.partial(<function _or_policy at 0x7ab0beb3af80>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ab0beb3ae60>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
(WorkerDict pid=689549) Total steps: 111981, num_warmup_steps: 0
(WorkerDict pid=689549) Actor use_remove_padding=True
(WorkerDict pid=689549) Actor use_fused_kernels=False
(WorkerDict pid=689739) Critic use_remove_padding=True
(WorkerDict pid=689549) /usr/local/lib/python3.10/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:690: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
(WorkerDict pid=689549)   warnings.warn(
(WorkerDict pid=689739) Some weights of Qwen2ForTokenClassification were not initialized from the model checkpoint at ./models/merged_sft_model and are newly initialized: ['score.bias', 'score.weight']
(WorkerDict pid=689739) You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
(WorkerDict pid=689739) Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
(AsyncvLLMServer pid=690114) FastAPI listen on 192.168.100.8:52907
(AsyncvLLMServer pid=690114) override_generation_config: {'n': 1, 'logprobs': 0, 'repetition_penalty': 1.0, 'max_new_tokens': 4096, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
(WorkerDict pid=689739) actor_module de local_path ./models/merged_sft_model
(WorkerDict pid=689739) Skipping monkey patch for Qwen2ForCausalLM as use_fused_kernels is False or fused_kernels_backend is torch
(WorkerDict pid=689739) Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
(AsyncvLLMServer pid=690114) WARNING 08-05 12:10:00 [arg_utils.py:1663] Detected VLLM_USE_V1=1 with Engine in background thread. Usage should be considered experimental. Please report any issues on Github.
(AsyncvLLMServer pid=690114) WARNING 08-05 12:10:00 [cuda.py:93] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
(AsyncvLLMServer pid=690115) FastAPI listen on 192.168.100.8:56473
(AsyncvLLMServer pid=690115) override_generation_config: {'n': 1, 'logprobs': 0, 'repetition_penalty': 1.0, 'max_new_tokens': 4096, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
(AsyncvLLMServer pid=690114) instance_id: e6e4c155-94cd-490a-9fe5-090cb9b97a4a:9aMJ77:2:0 initializes with external actors: ['9aMJ77WorkerDict_0:0']
(AsyncvLLMServer pid=690114) VERL_VLLM_ZMQ_ADDRESSES: ['ipc:///tmp/verl_vllm_zmq_689549_root.ipc']
(AsyncvLLMServer pid=690114) Using blocking ray.get inside async actor. This blocks the event loop. Please use `await` on object ref with asyncio.gather if you want to yield execution to the event loop instead.
(WorkerDict pid=689739) /usr/local/lib/python3.10/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:690: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
(WorkerDict pid=689739)   warnings.warn(
(WorkerDict pid=689549) WARNING 08-05 12:10:12 [utils.py:2522] Methods determine_num_available_blocks,device_config,get_cache_block_size_bytes,initialize_cache not implemented in <vllm.v1.worker.gpu_worker.Worker object at 0x7aaec40e01c0>
(AsyncvLLMServer pid=690115) WARNING 08-05 12:10:01 [arg_utils.py:1663] Detected VLLM_USE_V1=1 with Engine in background thread. Usage should be considered experimental. Please report any issues on Github.
(AsyncvLLMServer pid=690115) WARNING 08-05 12:10:01 [cuda.py:93] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
(AsyncvLLMServer pid=690115) instance_id: e6e4c155-94cd-490a-9fe5-090cb9b97a4a:9aMJ77:2:1 initializes with external actors: ['9aMJ77WorkerDict_0:1']
(AsyncvLLMServer pid=690115) VERL_VLLM_ZMQ_ADDRESSES: ['ipc:///tmp/verl_vllm_zmq_689739_root.ipc']
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
(AsyncvLLMServer pid=690115) Using blocking ray.get inside async actor. This blocks the event loop. Please use `await` on object ref with asyncio.gather if you want to yield execution to the event loop instead.
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.72it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.72it/s]
(WorkerDict pid=689549)
(AsyncvLLMServer pid=690115) WARNING 08-05 12:10:16 [config.py:1239] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.

```
查看sandbox日记，发现在调用工具
```
 running command python /tmp/tmph92m_9v6/tmpngrsk30o.py [sandbox.runners.base]
2025-08-05 05:22:27 [debug    ] stop running command python /tmp/tmph92m_9v6/tmpngrsk30o.py [sandbox.runners.base]
2025-08-05 05:22:28 [debug    ] start processing python request with code ```
# Testing zero value for discriminant
result = -a_over_b * (sum(r1_plus_r2_plus_r3) + c1 * (-s2))
pr
```

# 模型保存
```
du -sh checkpoint/qwen2.5-05b_ppo/global_step_3/*
2.4G	checkpoint/qwen2.5-05b_ppo/global_step_3/actor
5.6G	checkpoint/qwen2.5-05b_ppo/global_step_3/critic
4.0K	checkpoint/qwen2.5-05b_ppo/global_step_3/data.pt
ls -alhtR checkpoint/qwen2.5-05b_ppo/global_step_3/
checkpoint/qwen2.5-05b_ppo/global_step_3/:
总计 20K
drwxr-xr-x 6 root root 4.0K Aug  5 20:21 ..
drwxr-xr-x 4 root root 4.0K Aug  5 20:20 .
-rw-r--r-- 1 root root 1.5K Aug  5 20:20 data.pt
drwxr-xr-x 3 root root 4.0K Aug  5 20:20 critic
drwxr-xr-x 3 root root 4.0K Aug  5 20:19 actor

checkpoint/qwen2.5-05b_ppo/global_step_3/critic:
总计 5.6G
drwxr-xr-x 4 root root 4.0K Aug  5 20:20 ..
-rw-r--r-- 1 root root   46 Aug  5 20:20 fsdp_config.json
drwxr-xr-x 3 root root 4.0K Aug  5 20:20 .
drwxr-xr-x 2 root root 4.0K Aug  5 20:20 huggingface
-rw-r--r-- 1 root root  15K Aug  5 20:20 extra_state_world_size_2_rank_1.pt
-rw-r--r-- 1 root root 1.9G Aug  5 20:20 optim_world_size_2_rank_1.pt
-rw-r--r-- 1 root root  15K Aug  5 20:20 extra_state_world_size_2_rank_0.pt
-rw-r--r-- 1 root root 1.9G Aug  5 20:20 optim_world_size_2_rank_0.pt
-rw-r--r-- 1 root root 943M Aug  5 20:19 model_world_size_2_rank_1.pt
-rw-r--r-- 1 root root 943M Aug  5 20:19 model_world_size_2_rank_0.pt

checkpoint/qwen2.5-05b_ppo/global_step_3/critic/huggingface:
总计 16M
drwxr-xr-x 3 root root 4.0K Aug  5 20:20 ..
drwxr-xr-x 2 root root 4.0K Aug  5 20:20 .
-rw-r--r-- 1 root root  11M Aug  5 20:20 tokenizer.json
-rw-r--r-- 1 root root 1.6M Aug  5 20:20 merges.txt
-rw-r--r-- 1 root root 2.7M Aug  5 20:20 vocab.json
-rw-r--r-- 1 root root  605 Aug  5 20:20 added_tokens.json
-rw-r--r-- 1 root root  885 Aug  5 20:20 config.json
-rw-r--r-- 1 root root  613 Aug  5 20:20 special_tokens_map.json
-rw-r--r-- 1 root root 7.2K Aug  5 20:20 tokenizer_config.json

checkpoint/qwen2.5-05b_ppo/global_step_3/actor:
总计 2.4G
drwxr-xr-x 4 root root 4.0K Aug  5 20:20 ..
drwxr-xr-x 3 root root 4.0K Aug  5 20:19 .
-rw-r--r-- 1 root root  15K Aug  5 20:19 extra_state_world_size_2_rank_1.pt
-rw-r--r-- 1 root root 1.3K Aug  5 20:19 optim_world_size_2_rank_1.pt
-rw-r--r-- 1 root root 1.2G Aug  5 20:19 model_world_size_2_rank_1.pt
-rw-r--r-- 1 root root   46 Aug  5 20:19 fsdp_config.json
drwxr-xr-x 2 root root 4.0K Aug  5 20:19 huggingface
-rw-r--r-- 1 root root  15K Aug  5 20:19 extra_state_world_size_2_rank_0.pt
-rw-r--r-- 1 root root 1.2G Aug  5 20:19 model_world_size_2_rank_0.pt
-rw-r--r-- 1 root root 1.3K Aug  5 20:19 optim_world_size_2_rank_0.pt

checkpoint/qwen2.5-05b_ppo/global_step_3/actor/huggingface:
总计 16M
drwxr-xr-x 3 root root 4.0K Aug  5 20:19 ..
drwxr-xr-x 2 root root 4.0K Aug  5 20:19 .
-rw-r--r-- 1 root root  11M Aug  5 20:19 tokenizer.json
-rw-r--r-- 1 root root 1.6M Aug  5 20:19 merges.txt
-rw-r--r-- 1 root root 2.7M Aug  5 20:19 vocab.json
-rw-r--r-- 1 root root  605 Aug  5 20:19 added_tokens.json
-rw-r--r-- 1 root root  613 Aug  5 20:19 special_tokens_map.json
-rw-r--r-- 1 root root 7.2K Aug  5 20:19 tokenizer_config.json
-rw-r--r-- 1 root root  722 Aug  5 20:19 config.json
-rw-r--r-- 1 root root  242 Aug  5 20:19 generation_config.json

```

**评估结果（250步）**：

* acc\@30: **0.55**
* 平均调用轮数：**8.3**

PPO 相比 GRPO 在该设置中略低，可能与超参或策略更新有关。

---

合并FSDP训练后ppo的actor模型
```
检查最后一个step输出模型:
ls checkpoint/qwen2.5-05b_ppo/global_step_4

cd checkpoint/qwen2.5-05b_ppo/global_step_4/actor/
cp -a huggingface/* .
cd - 
python /workspace/verl/verl/scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoint/qwen2.5-05b_ppo/global_step_4/actor/ \
    --target_dir checkpoint/merged_ppo_model
输出:
Got device mesh tensor([0, 1], dtype=torch.int32), mesh_dim_names ('fsdp',)
Processing model shards with 2 (2,) in total
Loading 2 FSDP shards: 100%|█████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.29it/s]
Sliding Window Attention is enabled but not implemented for `eager`; unexpected results may be encountered.
Saving model to checkpoint/merged_dapo_model
Saving tokenizer to checkpoint/merged_dapo_model

ls -alht checkpoint/merged_ppo_model
total 1.2G
drwxr-xr-x 2 root root 4.0K Aug  5 12:25 .
-rw-r--r-- 1 root root  11M Aug  5 12:25 tokenizer.json
-rw-r--r-- 1 root root 1.6M Aug  5 12:25 merges.txt
-rw-r--r-- 1 root root 2.7M Aug  5 12:25 vocab.json
-rw-r--r-- 1 root root  605 Aug  5 12:25 added_tokens.json
-rw-r--r-- 1 root root  613 Aug  5 12:25 special_tokens_map.json
-rw-r--r-- 1 root root 7.2K Aug  5 12:25 tokenizer_config.json
-rw-r--r-- 1 root root 1.2G Aug  5 12:25 model.safetensors
-rw-r--r-- 1 root root  683 Aug  5 12:25 config.json
-rw-r--r-- 1 root root  242 Aug  5 12:25 generation_config.json
drwxr-xr-x 7 root root 4.0K Aug  5 12:25 ..
```


# 推理训练后的模型
```
# 使用哪个模型
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
ls checkpoint/merged_ppo_model
vllm serve checkpoint/merged_ppo_model --host 0.0.0.0 --port 5306
输出：

测试是否获取模型成功
# curl http://localhost:5306/v1/models
{"object":"list","data":[{"id":"checkpoint/merged_ppo_model","object":"model","created":1754396952,"owned_by":"vllm","root":"checkpoint/merged_ppo_model","parent":null,"max_model_len":32768,"permission":[{"id":"modelperm-cc4c9778969b4ecba3e4f5485ecc9fa5","object":"model_permission","created":1754396952,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}(base)
# 测试一条数据
curl http://localhost:5306/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"checkpoint/merged_ppo_model","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\""}]}'
输出:
{"id":"chatcmpl-13953aae28754cf78dea439c47e40e64","object":"chat.completion","created":1754396973,"model":"checkpoint/merged_ppo_model","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"To solve the given problem, we need to find the sum \\( AB + AC \\), which is equivalent to \\( \\frac{a\\sqrt{b}}{c} \\) with \\( a + b + c = 8 \\). The steps involve using trigonometric identities, coordinate geometry, and solving for specific ratios.\n  \nFirst, assign coordinates to points \\( A \\), \\( D \\), and \\( C \\) on the plane. Assign \\( A \\) at \\( (0, 0) \\) and \\( B \\) at \\( (c, 0) \\). Since \\( \\sin \\angle A = \\frac{4}{5} \\), point \\( C \\) must lie on the circle centered at \\( O \\) with radius \\( \\frac{4}{5} \\). Using parametric equations, we derive \\( |AC| = 2 \\).\n  \nThe angle bisector from \\( A \\) divides the angle \\( \\angle A \\) into two equal parts. The angle bisector theorem applies here:\n\\[\n\\frac{AD}{DB} = \\frac{AC}{BC}\n\\]\nGiven \\( AD = 1 \\), this implies \\( DB = \\frac{1}{2} \\) and \\( BC = \\frac{\\sqrt{7}}{2} \\).\n\nNext, compute the area \\( S \\) of triangle \\( ABC \\):\n\\[\nS = \\frac{1}{2} \\times AB \\times AC \\times \\sin \\angle A = \\frac{1}{2} \\times 2c \\times 1 \\times \\frac{4}{5} = \\frac{4c}{5}\n\\]\n\nUsing the Law of Cosines in triangles \\( BDA \\) and \\( DCB \\), relate the sides and angles. Applying cosine rule:\n\\[\nBC^2 = BD^2 + CD^2 - 2 \\cdot BD \\cdot CD \\cdot \\cos(\\angle BDC)\n\\]\nSubstitute \\( BC = \\frac{\\sqrt{7}}{2} \\), \\( BD = \\frac{1}{2} \\), \\( CD = \\frac{3}{2} \\), and \\( \\cos(\\angle BDC) = \\frac{1}{2} \\):\n\\[\n\\left( \\frac{\\sqrt{7}}{2} \\right)^2 = \\left( \\frac{1}{2} \\right)^2 + \\left( \\frac{3}{2} \\right)^2 - 2 \\cdot \\frac{1}{2} \\cdot \\frac{3}{2} \\cdot \\frac{1}{2}\n\\]\nSimplify:\n\\[\n\\frac{7}{4} = \\frac{1}{4} + \\frac{9}{4} - \\frac{3}{8}\n\\]\nConvert all terms to common denominator 8:\n\\[\n\\frac{7}{4} = \\frac{2}{8} + \\frac{18}{8} - \\frac{3}{8} = \\frac{23}{8}\n\\]\nThus, \\( S = \\frac{4c}{5} = \\frac{23}{8} \\implies c = \\frac{26}{5} \\).\n\nVerification using exact expressions confirms \\( AB + AC = \\frac{26}{5} \\).\n\n\n**Final Answer**\n\n\\boxed{26/5}","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":207,"total_tokens":919,"completion_tokens":712,"prompt_tokens_details":null},"prompt_logprobs":null}
---

## 🧠 总结：你需要知道的核心信息

| 阶段       | 方法   | 数据集           | 脚本                      | 准确率（acc\@30） | 平均轮数 |
| -------- | ---- | ------------- | ----------------------- | ------------ | ---- |
| 微调 (SFT) | SFT  | ReTool-SFT    | `run_qwen2-32b_sft.sh`  | 0.24         | 7.2  |
| 强化学习     | GRPO | DAPO-Math-17k | `run_qwen2-32b_dapo.sh` | 0.6          | 10   |
| 强化学习     | PPO  | DAPO-Math-17k | `run_qwen2-32b_ppo.sh`  | 0.55         | 8.3  |

