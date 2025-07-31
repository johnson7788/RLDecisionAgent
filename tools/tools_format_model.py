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

text = tokenizer.apply_chat_template(
    messages,
    tools=tools_second,
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

