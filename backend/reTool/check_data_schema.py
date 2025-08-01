#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/1 21:45
# @File  : check_data_schema.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :


import pyarrow.parquet as pq

pq_file = pq.ParquetFile("dataset/BytedTsinghua/train.parquet")
print(f"Train 文件schema")
print(pq_file.schema)

print(f"Test 文件schema")
pq_file = pq.ParquetFile("dataset/Maxwell/test.parquet")
print(pq_file.schema)
