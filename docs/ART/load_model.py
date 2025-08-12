#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/12 20:26
# @File  : load_model.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import pipeline

pipe = pipeline("text-generation", model="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
messages = [
            {"role": "user", "content": "Who are you?"},
            ]
pipe(messages)