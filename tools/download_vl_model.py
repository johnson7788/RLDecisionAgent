#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/19 19:48
# @File  : download_vl_model.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :  下载VL模型

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

# path = "OpenGVLab/InternVL3_5-1B-HF"
path = "unsloth/gemma-3-4b-it"
model = AutoModelForImageTextToText.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto").eval()