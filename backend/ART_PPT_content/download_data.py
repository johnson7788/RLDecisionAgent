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

# 加载 train 切分（显式写 split 更稳妥）
ds = load_dataset("DavideTHU/chinese_news_dataset", split="train")

# 转 DataFrame
df = ds.to_pandas()

# 如果有 output 列则重命名为 content；若本来就叫 content 则跳过
if "output" in df.columns:
    df = df.rename(columns={"output": "content"})
elif "content" in df.columns:
    pass
else:
    raise KeyError("数据集中未找到 'output' 或 'content' 字段，请检查列名。")

# 只保留需要的两列（若缺失会抛错，方便尽早发现问题）
df = df[["url", "content"]]

# 导出为 JSONL（utf-8，不转义中文）
df.to_json("./mcp_search/train_url_content.jsonl", orient="records", lines=True, force_ascii=False)

print("已导出到 ./mcp_search/train_url_content.jsonl")


