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

### 2. 启动训练脚本， 共用了约 12 分钟多时间。

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

## 🔁 强化学习阶段（RL）

Retool 提供了两种 RL 策略：

### 🎯 GRPO（Generalized REINFORCE with Policy Optimization）

```bash
注意修改：model_path，即SFT的训练后的模型结果
bash run_qwen2-05b_dapo.sh
```

**评估结果（150步）**：

* acc\@30: **0.6**
* 平均调用轮数：**10**

说明 RL 后模型能更灵活使用工具，提升了准确率。

---

### 🤖 PPO（Proximal Policy Optimization）

```bash
bash recipe/retool/run_qwen2-05b_ppo.sh
```

**评估结果（250步）**：

* acc\@30: **0.55**
* 平均调用轮数：**8.3**

PPO 相比 GRPO 在该设置中略低，可能与超参或策略更新有关。

---

## 🧠 总结：你需要知道的核心信息

| 阶段       | 方法   | 数据集           | 脚本                      | 准确率（acc\@30） | 平均轮数 |
| -------- | ---- | ------------- | ----------------------- | ------------ | ---- |
| 微调 (SFT) | SFT  | ReTool-SFT    | `run_qwen2-32b_sft.sh`  | 0.24         | 7.2  |
| 强化学习     | GRPO | DAPO-Math-17k | `run_qwen2-32b_dapo.sh` | 0.6          | 10   |
| 强化学习     | PPO  | DAPO-Math-17k | `run_qwen2-32b_ppo.sh`  | 0.55         | 8.3  |

