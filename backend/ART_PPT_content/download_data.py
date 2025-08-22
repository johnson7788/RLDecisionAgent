#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/22 14:11
# @File  : download_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 下载数据

# 1) 安装
# pip install -U datasets transformers accelerate

import pandas as pd
from datasets import load_dataset

# 2) 加载数据集
ds = load_dataset("DavideTHU/chinese_news_dataset")  # 默认取 train
print(ds)
print(ds["train"][0])
df = ds["train"].to_pandas()
head = df.head()
print(head)