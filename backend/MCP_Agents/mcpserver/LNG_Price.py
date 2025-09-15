#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/4/17
# @File  : lng_price_analysis.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 目前浙江地区LNG送到价格是多少？与历史同期对比高低？具体原因是什么？
# 目前山西地区LNG送到价格是多少？与历史同期对比高低？具体原因是什么？
# SOP, Standard Operating Procedure (SOP)
# 1. **识别地区与时间范围**：
#    - 从问题中提取出地区名称（如“浙江”、“山西”、“河北”）。
#    - 获取当前日期。
#    - 获取历史同期时间（通常为去年同月或同期）。
#
# 2. **查询当前价格与历史价格**：
#    - 查询该地区当前LNG送到价格。
#    - 查询该地区历史同期LNG送到价格。
#    - 对比当前价格与历史价格的差异，并说明是上涨、下跌还是持平。
#
# 3. **分析供需库存与市场信息**：
#    - 获取该地区最近的供需数据（需求量、供应量、库存情况）。
#    - 检查是否存在供需缺口或库存变动导致价格变动。
#    - 汇总最近的相关新闻（如天气变化、国际价格波动、政策调控等），用于补充原因分析。
#
# 4. **总结结论**：
#    - 结合价格变化、供需情况与新闻内容，给出简明扼要的总结，说明“价格变动的主要原因”。
#    - 若信息不足，可以说明“目前缺乏足够数据判断价格变动原因”。

from fastmcp import FastMCP
from typing import List, Dict
from datetime import datetime, timedelta
import random

# 创建 FastMCP 服务
mcp = FastMCP("LNG价格信息服务")

# 模拟数据：各地区LNG到岸价格（元/立方米）
SIMULATED_LNG_PRICES = {
    "浙江": {
        "2025-04": 4.5,
        "2025-03": 4.2,
        "2024-04": 4.0,
    },
    "山西": {
        "2025-04": 4.3,
        "2025-03": 4.1,
        "2024-04": 3.9,
    },
    "河北": {
        "2025-04": 4.4,
        "2025-03": 4.0,
        "2024-04": 3.8,
    }
}

# 模拟数据：供需及库存信息
SIMULATED_SUPPLY_DEMAND = {
    "浙江": [
        {"date": "2025-04-17", "demand": 1000, "supply": 900, "inventory": 5000},
        {"date": "2025-04-16", "demand": 950, "supply": 920, "inventory": 5100},
    ],
    "山西": [
        {"date": "2025-04-17", "demand": 800, "supply": 850, "inventory": 4200},
        {"date": "2025-04-16", "demand": 790, "supply": 800, "inventory": 4300},
    ],
    "河北": [
        {"date": "2025-04-17", "demand": 870, "supply": 840, "inventory": 4600},
        {"date": "2025-04-16", "demand": 860, "supply": 850, "inventory": 4550},
    ],
}

# 模拟新闻数据
SIMULATED_NEWS = [
    {"title": "浙江LNG需求激增", "date": "2025-04-15", "content": "由于工业用气增长，浙江地区LNG价格出现上涨。"},
    {"title": "全球LNG供应收紧", "date": "2025-04-10", "content": "进口量减少导致中国LNG整体供应趋紧，价格波动加剧。"},
    {"title": "山西工业回暖带动LNG需求", "date": "2025-04-14", "content": "多个重点工业园区恢复生产，推动LNG消耗上升。"},
    {"title": "河北调峰库存减少", "date": "2025-04-13", "content": "近期气温波动导致河北天然气库存使用增加，库存水平下降。"},
]


@mcp.prompt("SOP")
def sop_lng_price_analysis(question: str) -> str:
    """
    返回AI在面对关于“某地LNG送到价格及其变化原因”的提问时，应遵循的标准分析步骤（SOP）。
    输入参数：
        question: 用户提出的问题，例如“目前浙江地区LNG送到价格是多少？与历史同期对比高低？具体原因是什么？”
    返回：
        分析该类问题的标准操作流程（SOP），供AI使用以回答类似问题。
    """
    return f"""
你需要回答一个关于液化天然气（LNG）价格变动的问题，以下是标准的思考与分析流程（SOP）：

1. **识别地区与时间范围**：
   - 从问题中提取出地区名称（如“浙江”、“山西”、“河北”）。
   - 获取当前日期。
   - 获取历史同期时间（通常为去年同月或同期）。

2. **查询当前价格与历史价格**：
   - 查询该地区当前LNG送到价格。
   - 查询该地区历史同期LNG送到价格。
   - 对比当前价格与历史价格的差异，并说明是上涨、下跌还是持平。

3. **分析供需库存与市场信息**：
   - 获取该地区最近的供需数据（需求量、供应量、库存情况）。
   - 检查是否存在供需缺口或库存变动导致价格变动。
   - 汇总最近的相关新闻（如天气变化、国际价格波动、政策调控等），用于补充原因分析。

4. **总结结论**：
   - 结合价格变化、供需情况与新闻内容，给出简明扼要的总结，说明“价格变动的主要原因”。
   - 若信息不足，可以说明“目前缺乏足够数据判断价格变动原因”。

请根据以上步骤，分析以下问题并回答：
【{question}】
"""

@mcp.tool()
def get_current_date() -> str:
    """返回今天的日期，格式为YYYY-MM-DD。"""
    return datetime.now().strftime("%Y-%m-%d")


@mcp.tool()
def get_lng_price(region: str, start_date: str, end_date: str = None) -> List[Dict]:
    """
    获取指定地区从start_date到end_date之间的每日LNG到岸价格。
    若未指定end_date，则默认为今天。
    region:
    start_date: 2020-02-03
    """
    print(f"获取{region}地区从{start_date}到{end_date}之间的每日LNG到岸价格...")
    region = region.lower()
    if region not in SIMULATED_LNG_PRICES:
        return []

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.now() if not end_date else datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return []

    prices = []
    current = start
    while current <= end:
        month_key = current.strftime("%Y-%m")
        base_price = SIMULATED_LNG_PRICES[region].get(month_key, 4.0)
        daily_price = base_price + random.uniform(-0.05, 0.05)  # 添加轻微波动
        prices.append({
            "date": current.strftime("%Y-%m-%d"),
            "price": round(daily_price, 2)
        })
        current += timedelta(days=1)

    return prices

@mcp.tool()
def get_supply_demand_news(region: str, days: int = 15) -> Dict:
    """
    获取指定地区的供需情况及最近的相关新闻。支持浙江，山西，河北
    """
    print(f"获取{region}地区最近{days}天的供需情况及相关新闻...")
    region = region.lower()
    if region not in SIMULATED_SUPPLY_DEMAND:
        return {"supply_demand": [], "news": []}

    supply_demand = SIMULATED_SUPPLY_DEMAND.get(region, [])

    cutoff_date = datetime.now() - timedelta(days=days)
    news = [
        n for n in SIMULATED_NEWS
        if datetime.strptime(n["date"], "%Y-%m-%d") >= cutoff_date and region in n["content"]
    ]

    return {
        "supply_demand": supply_demand,
        "news": news
    }

if __name__ == '__main__':
    get_supply_demand_news("浙江", 15)