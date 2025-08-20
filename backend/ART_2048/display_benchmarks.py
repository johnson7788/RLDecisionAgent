#!/usr/bin/env python
# coding: utf-8
"""
基准评估脚本：
1. 从 S3 或本地加载训练数据（trajectories）
2. 对比多个模型（训练模型 & GPT 系列基线）
3. 绘制训练进度曲线图 (Win Rate over Time)
4. 绘制最终胜率柱状图 (Percentage of Games Won)
"""

# ==============================
# 1. 导入依赖
# ==============================
import importlib
import os
import asyncio
from dotenv import load_dotenv

# ART 提供的可视化与基准测试工具
import art.utils.benchmarking.charts
from art.utils.benchmarking.charts import (
    percentage_comparison_bar_chart,   # 绘制对比柱状图
    training_progress_chart,           # 绘制训练进度折线图
)
from art.utils.benchmarking.load_trajectories import load_trajectories
from art.utils.benchmarking.types import BenchmarkModelKey
from art.utils.get_repo_root_path import get_repo_root_path
from art.utils.s3 import pull_model_from_s3

# 热加载 charts 模块，便于 notebook 里调试
importlib.reload(art.utils.benchmarking.charts)

load_dotenv()

async def main():
    # ==============================
    # 2. 配置项目
    # ==============================
    project_name = "2048"

    # 是否从 S3 拉取模型（默认 False）
    PULL_MODELS = True
    if PULL_MODELS:
        # 拉取训练过的模型以及基线模型
        await pull_model_from_s3(model_name="003", project=project_name)
        await pull_model_from_s3(model_name="gpt-4o", project=project_name)
        await pull_model_from_s3(model_name="gpt-4.1", project=project_name)
        await pull_model_from_s3(model_name="gpt-4o-mini", project=project_name)

    # ==============================
    # 3. 加载 Trajectory 数据
    # ==============================
    # bust_cache() 可清理缓存，确保加载最新结果
    # await load_trajectories.bust_cache()

    df = await load_trajectories(
        project_name=project_name,
        models=["tutorial-001", "gpt-4o", "gpt-4.1", "gpt-4o-mini"],
    )

    # 输出图表存储路径
    benchmarks_dir = f"{get_repo_root_path()}/assets/benchmarks/{project_name}"
    os.makedirs(benchmarks_dir, exist_ok=True)


    # ==============================
    # 4. 绘制训练进度曲线图
    # ==============================
    line_graph = training_progress_chart(
        df,
        metric_name="win",   # 指标：胜率
        models=[
            BenchmarkModelKey("tutorial-001", "tutorial-001", "train"),  # 自己训练的模型
            BenchmarkModelKey("gpt-4o", "GPT-4o"),                      # GPT-4o 基线
            BenchmarkModelKey("gpt-4.1", "GPT-4.1"),                    # GPT-4.1 基线
            BenchmarkModelKey("gpt-4o-mini", "GPT-4o-mini"),            # GPT-4o-mini 基线
        ],
        title="Win Rate over Time",
        y_label="Win Rate",
    )
    line_graph.savefig(f"{benchmarks_dir}/accuracy-training-progress.svg")


    # ==============================
    # 5. 绘制最终胜率柱状图
    # ==============================
    bar_chart = percentage_comparison_bar_chart(
        df,
        metric_name="win",
        models=[
            BenchmarkModelKey("tutorial-001", "tutorial-001", "train"),
            BenchmarkModelKey("gpt-4o", "GPT-4o"),
            BenchmarkModelKey("gpt-4.1", "GPT-4.1"),
            BenchmarkModelKey("gpt-4o-mini", "GPT-4o-mini"),
        ],
        title="Percentage of Games Won",
    )
    bar_chart.savefig(f"{benchmarks_dir}/accuracy-comparison.svg")

if __name__ == '__main__':
    asyncio.run(main())