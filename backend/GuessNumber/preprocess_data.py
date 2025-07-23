#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/7/19 02:57
# @File  : prepare_data.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 预处理数据


import argparse
import os

import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="guess_data")
    args = parser.parse_args()

    # 从本地 dataset.json 读取对话数据
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    print(f"正在从 {dataset_path} 读取对话数据...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"共读取到 {len(conversations)} 条对话数据。")

    # Create train and test datasets
    split_idx = int(len(conversations) * 0.8)
    print(f"将数据集切分为 {split_idx} 条训练数据和 {len(conversations) - split_idx} 条测试数据。")
    train_data = conversations[:split_idx]  # 前80%用于训练
    test_data = conversations[split_idx:]   # 后20%用于测试

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    print(f"输出目录为: {local_dir}")

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    print(f"训练集已保存到: {train_path}")
    print(f"测试集已保存到: {test_path}")

    # Print statistics
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()
