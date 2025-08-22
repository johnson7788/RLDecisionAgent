"""
本模块用于：
1) 从 HuggingFace 拉取 Hacker News 数据集并做简单筛选；
2) 将故事(story)序列化为统一字符串格式，便于下游模型评分/生成标题；
3) 计算按数据集 split 划分的评估指标（RMSE、相关系数）；
4) 通过异步 HTTP 请求调用“奖励模型（reward model）”为标题打分，并用 SQLite 做结果缓存；
5) 构造生成标题的对话提示（prompt）。

依赖：
- httpx（异步 HTTP 客户端）
- polars（高性能 DataFrame）
- datasets（HuggingFace Datasets）
- pydantic（数据校验/序列化）
- panza.SQLiteCache（简单的函数级缓存）
- python-dotenv（加载 .env 环境变量）

注意：
- 代码中对 `cache_db_path` / `cache` 有一次重复定义，如非必要可以保留一次即可。
"""

import math
import os
from datetime import datetime
from typing import Dict, Optional

import httpx
import polars as pl
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from panza import SQLiteCache
from pydantic import BaseModel, Field, field_serializer

# 从 .env 文件加载环境变量，例如OPENAI的key
load_dotenv()

# 缓存数据库文件路径（与当前文件同目录下）
cache_db_path = os.path.join(os.path.dirname(__file__), "shared_cache.db")
# 初始化 SQLite 缓存，用于给函数结果做持久化缓存
cache = SQLiteCache(cache_db_path)


class ScoreRequest(BaseModel):
    """
    用于发送到奖励模型(reward model)的请求体数据结构。
    Pydantic 会负责字段校验与序列化。
    """
    title: str = Field(..., description="The title of the story")
    by: str = Field(..., description="The submitter of the story")
    time: str = Field(..., description="The submission time of the story")
    scraped_body: str = Field(..., description="The body content of the story")
    url: Optional[str] = Field(None, description="The URL of the story")

    @field_serializer("time")
    def serialize_time(self, value: datetime) -> str:
        """
        将时间字段统一序列化为 ISO 8601 字符串。
        兼容传入 str 或 datetime 两种情况（若已是 str 则直接返回）。
        """
        if isinstance(value, str):
            return value
        return value.isoformat()


def serialize_story(story):
    """
    将单条 story（字典）序列化为带标签的长字符串，便于 LLM/评分模型消费。
    字段包括：提交者、URL、日期、正文、标题。
    其中日期格式化为 YYYY-MM-DD。
    """
    string = f"""<submitter>{story["by"]}</submitter>\n<url>{story["url"]}</url>\n<date>{story["time"].strftime("%Y-%m-%d")}</date>\n\n<body>{story["scraped_body"]}</body>\n<title>{story["title"]}</title>"""
    return string


def with_serialized_stories(df: pl.DataFrame) -> pl.DataFrame:
    """
    在现有 Polars DataFrame 上新增一列 `serialized`，其值为每行 story 的序列化文本。
    这里使用 pl.struct 组合多列为单个结构，再用 map_elements 应用 Python 函数。
    """
    return df.with_columns(
        pl.struct(["title", "by", "time", "scraped_body", "url"])
        .map_elements(serialize_story, return_dtype=pl.Utf8)  # 指定返回字符串类型
        .alias("serialized")
    )


def calculate_metrics_by_split(df: pl.DataFrame) -> pl.DataFrame:
    """
    计算数据集中每个 split 的相关系数(correlation)与 RMSE 指标。

    参数:
        df: 包含列 log_score（真值）、predictions（模型预测）、split（数据划分）的 DataFrame

    返回:
        每个 split 一行的指标表，包括 baseline_rmse（以均值为基线）、model_rmse、model_correlation
    """
    metrics = []

    # 遍历每个 split（如 train/validation/test）
    for split in df["split"].unique():
        split_df = df.filter(pl.col("split") == split)

        # —— 基线（以均值作为预测）RMSE ——
        average_score = split_df["log_score"].mean()
        rmse_baseline = math.sqrt(
            (split_df["log_score"] - average_score).pow(2).sum() / len(split_df)
        )

        # —— 模型预测的 RMSE 与皮尔逊相关系数 ——
        rmse_model = math.sqrt(
            (split_df["log_score"] - split_df["predictions"]).pow(2).sum()
            / len(split_df)
        )
        # pl.corr 返回单值列，这里取出第一个元素
        correlation_model = split_df.select(pl.corr("log_score", "predictions"))[
            "log_score"
        ][0]

        metrics.append(
            {
                "split": split,
                "baseline_rmse": rmse_baseline,
                "model_rmse": rmse_model,
                "model_correlation": correlation_model,
            }
        )

    return pl.DataFrame(metrics)


# 奖励模型服务的 URL，优先从环境变量 REWARD_MODEL_URL 读取，否则用默认值
REWARD_MODEL_URL = os.getenv(
    "REWARD_MODEL_URL", "https://openpipe-dev--hn-title-rm-serve-rm.modal.run/score"
)

print(f"使用的奖励模型服务 URL: {REWARD_MODEL_URL}")


@cache.cache()  # 使用 SQLite 缓存函数输出，避免对同一输入重复请求远端服务
async def score_title(
    story_dict: Dict,
    _reward_model: str = REWARD_MODEL_URL,
) -> float:
    """异步调用奖励模型，为给定 story 计算得分。

    参数:
        story_dict: 包含 title, by, time, scraped_body, url 的字典
        _reward_model: 奖励模型标识/URL

    返回:
        奖励模型返回的分数（float）。若请求超时/异常，返回 0.0 作为默认值。
    """
    # 设置较长超时时间以防模型推理时间过长
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            # 拷贝原始 dict，避免就地修改
            request_dict = story_dict.copy()
            # 将时间字段规范为 ISO 字符串（以防传入的是 datetime）
            request_dict["time"] = request_dict["time"].isoformat()
            # 通过 Pydantic 模型再做一层校验与序列化
            response = await client.post(
                REWARD_MODEL_URL, json=ScoreRequest(**request_dict).model_dump()
            )
            response.raise_for_status()
            data = response.json()
            return data["score"]
        except httpx.TimeoutException:
            # 超时：打印告警并返回缺省分数
            print(f"Timeout connecting to reward model at {REWARD_MODEL_URL}")
            return 0.0
        except Exception as e:
            # 其他异常：打印错误信息并返回缺省分数
            print(f"Error connecting to reward model: {str(e)}")
            return 0.0


# —— 下方两行与上文重复，如无特定需要，可删除其一以免混淆 ——
cache_db_path = os.path.join(os.path.dirname(__file__), "shared_cache.db")
cache = SQLiteCache(cache_db_path)


def pull_data(
    split: str = "train",
    max_items: int = 10,
    min_score: int = 20,
) -> Dataset:
    """
    从 HuggingFace 拉取 Hacker News 数据集，并按分数与数量做筛选采样。

    参数：
        split: 数据集划分（如 "train" / "validation" / "test"）
        max_items: 取前多少条样本（用于快速实验）
        min_score: 仅保留 score >= min_score 的样本

    返回：
        处理后的 Dataset 对象
    """
    print(f"从 HuggingFace 下载数据集hacker-news-scraped-stories-filtered (max {max_items} items)...")
    dataset: Dataset = load_dataset(
        "OpenPipe/hacker-news-scraped-stories-filtered", split=split
    )  # type: ignore
    # 过滤低分样本
    dataset = dataset.filter(lambda x: x["score"] >= min_score)
    # 截取前 max_items 条
    dataset = dataset.select(range(max_items))
    return dataset


def prompt_for_title(content: str) -> list[dict]:
    """
    构造用于生成 HN 标题的对话消息（system + user）。
    要求模型仅输出“标题”本身，不包含多余文本。

    参数：
        content: HN 帖子正文或摘要，将作为提示中的“Content”部分

    返回：
        适配 Chat Completions 风格的消息列表（list[dict]）
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates engaging titles for Hacker News posts. Respond with just the title, no other text.",
        },
        {
            "role": "user",
            "content": f"Generate a concise, engaging title for this Hacker News submission. The title should be informative yet catchy.\n\nContent: {content}",
        },
    ]
