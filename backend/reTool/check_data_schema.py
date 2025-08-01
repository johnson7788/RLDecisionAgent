#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/1 21:45
# @File  : check_data_schema.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

from datasets import load_dataset
import pyarrow.parquet as pq

train_pq_file = pq.ParquetFile("dataset/BytedTsinghua/train/data.parquet")
print(f"Train 文件schema")
print(train_pq_file.schema)
train_table = train_pq_file.read()
train_df = train_table.to_pandas()
print("Train DataFrame:")
print(train_df.head())

print(f"Test 文件schema")
test_pq_file = pq.ParquetFile("dataset/Maxwell/validation/data.parquet")
print(test_pq_file.schema)
test_table = test_pq_file.read()
test_df = test_table.to_pandas()
print("Test DataFrame:")
print(test_df.head())


parquet_file = "dataset/BytedTsinghua/train/data.parquet"

# 正确方式
train_dataset = load_dataset("parquet", data_files=parquet_file)["train"]
print(train_dataset[0])


parquet_file = "dataset/Maxwell/validation/data.parquet"

# 正确方式
train_dataset = load_dataset("parquet", data_files=parquet_file)["train"]
print(train_dataset[0])