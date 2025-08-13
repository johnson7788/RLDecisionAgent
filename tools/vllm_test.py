#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/13 11:40
# @File  : vllm_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 用一个最小的模型，测试下vllm是否安装正常

from vllm import LLM, SamplingParams

# 输入几个问题
prompts = [
    "你好",
    "Hi，How Are you ",
]

# 设置初始化采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# 加载模型，确保路径正确, arnir0/Tiny-LLM
llm = LLM(model="Tiny-LLM", trust_remote_code=True, max_model_len=1024)

# 进行推理
responses = llm.generate(prompts, sampling_params)

# 输出结果
for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response['text']}\n")
