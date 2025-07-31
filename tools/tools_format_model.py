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
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

# ✅ 构造对话内容
messages = [
    {"role": "user", "content": "请用Python计算根号下9384 + 3的9次方是多少？"}
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

# ✅ 模板构建与 tokenization
inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=True,
).to(model.device)

# ✅ 模型生成
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.7,
)

# ✅ 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🧠 模型响应内容：\n")
print(response)
