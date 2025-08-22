#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/23 07:24
# @File  : check_dataset.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 查看数据

# pip install datasets pandas pyarrow matplotlib tldextract
from typing import Tuple, List, Optional
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import tldextract
import math
import random
import matplotlib.pyplot as plt

def load_hn_dataset(max_items: int = 1000,
                    min_score: int = 0,
                    split: str = "train",
                    shuffle: bool = False,
                    seed: int = 42) -> Dataset:
    print(f"从 HuggingFace 下载数据集 OpenPipe/hacker-news-scraped-stories-filtered（最多取 {max_items} 条，分割={split}）...")
    ds: Dataset = load_dataset("OpenPipe/hacker-news-scraped-stories-filtered", split=split)  # type: ignore
    # 过滤低分样本
    if "score" in ds.column_names:
        ds = ds.filter(lambda x: (x.get("score", -math.inf) or -math.inf) >= min_score)
    else:
        print("⚠️ 数据集中不含列 'score'，跳过基于分数的过滤。")

    # 仅保留前 max_items 条
    if max_items is not None and max_items > 0:
        ds = ds.select(range(min(max_items, ds.num_rows)))

    if shuffle:
        ds = ds.shuffle(seed=seed)

    print(f"✅ 载入完成：{ds.num_rows} 行，{len(ds.column_names)} 列。列名：{ds.column_names}")
    return ds

def dataset_overview(ds: Dataset, show_examples: int = 5) -> None:
    print("\n=== 字段与类型（datasets.Features）===")
    try:
        for k, v in ds.features.items():
            print(f"- {k}: {v}")
    except Exception:
        # 有些数据集 features 可能缺失，做个兜底
        for k in ds.column_names:
            print(f"- {k}")

    print("\n=== 缺失值统计（基于前 N 行估计）===")
    # 为避免一次性转全量 Pandas（可能内存大），只用前 10k 行估计
    n = min(10_000, ds.num_rows)
    sample = ds.select(range(n)).to_pandas()
    miss = sample.isna().sum().sort_values(ascending=False)
    print(miss.to_string())

    print(f"\n=== 前 {show_examples} 条样本预览 ===")
    for i in range(min(show_examples, ds.num_rows)):
        row = ds[i]
        # 常见字段尽量展示；不存在则跳过
        title = row.get("title")
        url = row.get("url")
        by = row.get("by") or row.get("author")
        score = row.get("score")
        text = row.get("text")
        print(f"\n[{i}] title={title!r}\n    by={by} | score={score} | url={url}\n    text={(text[:160] + '...') if isinstance(text, str) and len(text) > 160 else text}")

def to_dataframe(ds: Dataset) -> pd.DataFrame:
    """将 Dataset 转为 Pandas DataFrame，便于进一步分析。"""
    df = ds.to_pandas()
    # 统一常见列名（若存在）
    if "by" in df.columns and "author" not in df.columns:
        df.rename(columns={"by": "author"}, inplace=True)
    return df

def extract_domain(url: Optional[str]) -> Optional[str]:
    if not isinstance(url, str) or not url:
        return None
    ext = tldextract.extract(url)
    if not ext.registered_domain:
        return None
    return ext.registered_domain  # 例如 'nytimes.com'

def quick_stats(df: pd.DataFrame) -> None:
    print("\n=== 数值列统计 ===")
    num_cols = [c for c in ["score", "descendants", "time"] if c in df.columns]
    if num_cols:
        print(df[num_cols].describe(percentiles=[.5, .9, .99]).to_string())
    else:
        print("（未检测到常见数值列）")

    # 文本长度
    if "title" in df.columns:
        title_len = df["title"].dropna().astype(str).str.len()
        print("\n标题长度：")
        print(title_len.describe(percentiles=[.5, .9, .99]).to_string())

    if "text" in df.columns:
        text_len = df["text"].dropna().astype(str).str.len()
        print("\n正文长度：")
        print(text_len.describe(percentiles=[.5, .9, .99]).to_string())

    # 作者 Top
    if "author" in df.columns:
        print("\n作者 Top 20：")
        print(df["author"].dropna().value_counts().head(20).to_string())

    # 域名 Top
    if "url" in df.columns:
        domains = df["url"].apply(extract_domain)
        print("\n外链域名 Top 20：")
        print(domains.dropna().value_counts().head(20).to_string())

    # 类型分布（若有）
    for col in ["type", "label"]:
        if col in df.columns:
            print(f"\n列 {col} 的值分布 Top 20：")
            print(df[col].dropna().value_counts().head(20).to_string())

def plot_distributions(df: pd.DataFrame) -> None:
    """可选：画两个最常用的分布图，便于快速感知数据形状。"""
    try:
        if "score" in df.columns:
            plt.figure()
            df["score"].dropna().astype(float).plot(kind="hist", bins=50, title="Score 分布")
            plt.xlabel("score")
            plt.ylabel("频次")
            plt.show()

        if "title" in df.columns:
            plt.figure()
            df["title"].dropna().astype(str).str.len().plot(kind="hist", bins=50, title="标题长度分布")
            plt.xlabel("title length")
            plt.ylabel("频次")
            plt.show()
    except Exception as e:
        print(f"⚠️ 绘图失败：{e}")

def explore_hn(max_items: int = 2000, min_score: int = 0, split: str = "train", draw_plots: bool = False):
    ds = load_hn_dataset(max_items=max_items, min_score=min_score, split=split)
    dataset_overview(ds, show_examples=5)
    print("\n=== 转为 Pandas 继续分析（可能会稍慢）===")
    df = to_dataframe(ds)
    quick_stats(df)
    if draw_plots:
        print("\n=== 绘图 ===")
        plot_distributions(df)
    return ds, df

if __name__ == "__main__":
    # 你可以按需修改这些参数
    ds, df = explore_hn(max_items=3000, min_score=50, split="train", draw_plots=True)
