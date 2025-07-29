# 使用强化学习方法（GRPO 和 PPO）训练大语言模型（Qwen2.5-32B），在数学推理任务中**策略性地使用工具**，以提升解题准确率。

# 文档
https://www.notion.so/verl-reTool-recipe-2398b5b7feba80a58156fa936f9f8de6

# 数据

下载和处理数据BytedTsinghua-SIA/DAPO-Math-17k
python3 examples/data_preprocess/dapo_multiturn_w_tool.py

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

## 🚀 微调阶段（SFT）

### 1. 数据预处理

```bash
python3 recipe/retool/retool_sft_preprocess.py
```

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


### 2. 启动训练脚本

```bash
bash recipe/retool/run_qwen2-32b_sft.sh
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
