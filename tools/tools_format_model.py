#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/7/31 13:14
# @File  : tools_format_model.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试模型使用工具的格式
# 输出示例
# 🧠 模型响应内容：
# <tool_call>
# {"name": "math.sqrt", "arguments": {"num": "9384"}}
# </tool_call>
# <tool_call>
# {"name": "pow", "arguments": {"base": "math.sqrt(9384)", "exponent": 9}}
# </tool_call>


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ 加载模型和 tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    sliding_window=None,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 构造对话内容
prompt = "请用Python计算根号下9384 + 3的9次方是多少？"
messages = [
    {"role": "system", "content": "You are Qwen,  You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# ✅ 构造工具 schema（注意 function schema 合法性）, 第一种tools格式和第2种tool格式，第一种比第2种更详细
tools_first = [
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "执行一段Python代码",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python代码"}
                },
                "required": ["code"]
            }
        }
    }
]

tools_second = [
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

messages = [{'content': "Let $\\mathbf{v}_0$ be a vector. The vector $\\mathbf{v}_0$ is projected onto $\\begin{pmatrix} 3 \\\\ 1 \\end{pmatrix},$ resulting in the vector $\\mathbf{v}_1.$ The vector $\\mathbf{v}_1$ is then projected onto $\\begin{pmatrix} 1 \\\\ 1 \\end{pmatrix},$ resulting in the vector $\\mathbf{v}_2.$ Find the matrix that takes $\\mathbf{v}_0$ to $\\mathbf{v}_2.$ The original answer is in the $format \\begin{pmatrix} \\frac{i_{11}}{5} & \\frac{i_{12}}{5} \\\\ \\frac{i_{21}}{5} & \\frac{i_{22}}{5} \\end{pmatrix}$, please give the value of $i_{11} + i_{12} + i_{21} + i_{22}$.\n\nRemember to place the final answer in the last part using the format: \n\n\\boxed{{'The final answer goes here.'}}", 'role': 'user', 'tool_calls': None},
 {'content': "Okay, so I need to find the matrix that takes a vector v0 and transforms it into v2, where v2 is the result of projecting v1 onto the vector (1, 1), and v1 is the projection of v0 onto (3, 1). Then, the answer should be a matrix with denominators 5, and I have to sum all the numerators to get the final answer. Let me break this down step by step.\n\nFirst, I remember that the projection of a vector a onto another vector b is given by the formula:\n\nproj_b a = ( (a · b) / ||b||² ) * b\n\nSo, let's denote the projection of v0 onto (3,1) as v1. Then, v1 = proj_{(3,1)} v0. Let's compute that.\n\nLet me represent v0 as a column vector (x, y). Then, the projection v1 is:\n\nv1 = ( [v0 · (3,1)] / ||(3,1)||² ) * (3,1)\n\nCompute the dot product: v0 · (3,1) = 3x + y.\n\nThe norm squared of (3,1) is 3² + 1² = 9 + 1 = 10.\n\nSo, v1 = ( (3x + y)/10 ) * (3,1) = ( (9x + 3y)/10, (3x + y)/10 )\n\nTo verify, I'll compute the projection matrix for the first step using code:", 'role': 'assistant', 'tool_calls': [{'function': {'arguments': {'code': 'import numpy as np\nb1 = np.array([3, 1])\nP1 = np.outer(b1, b1) / np.dot(b1, b1)\nprint("Projection matrix P1:\\n", P1)'}, 'name': 'code_interpreter'}, 'type': 'function'}]},
 {'content': 'Projection matrix P1:\n [[0.9 0.3]\n [0.3 0.1]]', 'role': 'tool', 'tool_calls': None},
 {'content': 'This confirms that the first projection matrix is indeed:\n\n\\[\nP1 = \\begin{pmatrix} \\frac{9}{10} & \\frac{3}{10} \\\\ \\frac{3}{10} & \\frac{1}{10} \\end{pmatrix}\n\\]\n\nNext, projecting v1 onto (1,1). The projection matrix for this step can be calculated as:', 'role': 'assistant', 'tool_calls': [{'function': {'arguments': {'code': 'b2 = np.array([1, 1])\nP2 = np.outer(b2, b2) / np.dot(b2, b2)\nprint("Projection matrix P2:\\n", P2)'}, 'name': 'code_interpreter'}, 'type': 'function'}]},
 {'content': 'Projection matrix P2:\n [[0.5 0.5]\n [0.5 0.5]]', 'role': 'tool', 'tool_calls': None},
 {'content': 'Sum of numerators: 8', 'role': 'tool', 'tool_calls': None},
 {'content': 'Therefore, the value of \\(i_{11} + i_{12} + i_{21} + i_{22}\\) is 8.\n\n\n\\boxed{8}', 'role': 'assistant', 'tool_calls': None}]

tools_third = [{"type": "function", "function": {"name": "code_interpreter", "description": "A tool for executing code.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to execute."}}, "required": ["code"]}}}]

text = tokenizer.apply_chat_template(
    messages,
    tools=tools_third,
    tokenize=False,
    add_generation_prompt=True
)

# ✅ 模板构建与 tokenization
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tools=tools,
#     tokenize=True,
#     return_tensors="pt",
#     add_generation_prompt=True,
#     enable_thinking=True,
# )


model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("🧠 模型响应内容：\n")
print(response)

