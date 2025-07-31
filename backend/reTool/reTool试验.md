# 使用强化学习方法（GRPO 和 PPO）训练大语言模型（Qwen2.5-32B），在数学推理任务中**策略性地使用工具**，以提升解题准确率。

# 文档
https://www.notion.so/verl-reTool-recipe-2398b5b7feba80a58156fa936f9f8de6

# 1. 数据下载

下载和处理数据BytedTsinghua-SIA/DAPO-Math-17k
export HF_ENDPOINT=https://hf-mirror.com
python3 examples/data_preprocess/dapo_multiturn_w_tool.py
```
输出
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████| 145/145 [00:00<00:00, 518kB/s]
dapo-math-17k.parquet: 100%|███████████████████████████████████████████████████████████████████████▉| 299M/299M [03:21<00:00, 1.48MB/s]
Generating train split: 1791700 examples [00:03, 449608.47 examples/s]
Map:  19%|██████████████▍                                                            | 345877/1791700 [00:46<03:03, 7857.80 examples/s]

```

数据会下载到~/data/retool_dapo
```
ls ~/data/retool_dapo
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

### 2.1 数据预处理

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

### 2. 启动训练脚本

```bash
注意设置使用哪个显卡
export CUDA_VISIBLE_DEVICES=1,2
bash run_qwen2-05b_sft.sh
```
* 启动基于 Qwen2.5-32B 的监督微调训练。

### ✅ 微调后评估结果

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
bash recipe/retool/run_qwen2-32b_dapo.sh
```

**评估结果（150步）**：

* acc\@30: **0.6**
* 平均调用轮数：**10**

说明 RL 后模型能更灵活使用工具，提升了准确率。

---

### 🤖 PPO（Proximal Policy Optimization）

```bash
bash recipe/retool/run_qwen2-32b_ppo.sh
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



## 错误总结
RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
ulysses_sequence_parallel_size
Ulysses 的序列并行维度 ulysses_sequence_parallel_size 超出了实际 GPU 数量，构建 device mesh 会失败。
CUDA_VISIBLE_DEVICES=1,2  # 实际只有 2 块卡
torchrun --nproc_per_node=2
改成 ulysses_sequence_parallel_size=2
