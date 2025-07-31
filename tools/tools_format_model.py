#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/7/31 13:14
# @File  : tools_format_model.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试模型使用工具的格式


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ 加载模型和 tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
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

# ✅ 构造工具 schema（注意 function schema 合法性）
tools = [
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

text = tokenizer.apply_chat_template(
    messages,
    tools=tools,
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
