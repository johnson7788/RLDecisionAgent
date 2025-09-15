#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/4/17 (Modified: 2025-04-22)
# @File  : factory_profit.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : eg: 查询内蒙古在某个时间段内的液厂利润水平

from fastmcp import FastMCP
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta, date
import random
import statistics # 用于计算平均值

mcp = FastMCP("LNG工厂利润计算服务")
# --- 模拟数据 ---

# 1. 模拟原料气竞拍价格 (元/立方米)
SIMULATED_AUCTION_PRICES: Dict[str, Dict[str, float]] = {
    "内蒙古": {},
    "河北": {}
}

# 2. 模拟省份对应的工厂列表
SIMULATED_FACTORIES: Dict[str, List[str]] = {
    "内蒙古": ["内蒙古工厂A", "内蒙古工厂B", "内蒙古工厂C"],
    "河北": ["河北工厂X", "河北工厂Y"]
}

# 3. 模拟各工厂的出厂价格 (元/吨)
SIMULATED_FACTORY_PRICES: Dict[str, Dict[str, float]] = {}

# --- 动态生成模拟数据 (例如，生成最近30天的数据) ---
today = datetime.now().date() # 使用 date 对象更方便比较
for i in range(30): # 生成包括今天在内的最近30天数据
    current_date_obj = today - timedelta(days=i)
    date_str = current_date_obj.strftime("%Y-%m-%d")

    # 内蒙古竞拍价模拟 (假设在 2.8 到 3.2 之间波动)
    SIMULATED_AUCTION_PRICES["内蒙古"][date_str] = round(random.uniform(2.8, 3.2), 2)
    # 河北竞拍价模拟 (假设在 3.0 到 3.4 之间波动)
    SIMULATED_AUCTION_PRICES["河北"][date_str] = round(random.uniform(3.0, 3.4), 2)

    # 内蒙古工厂价格模拟 (假设在 5800 到 6200 元/吨之间波动)
    for factory in SIMULATED_FACTORIES["内蒙古"]:
        if factory not in SIMULATED_FACTORY_PRICES:
            SIMULATED_FACTORY_PRICES[factory] = {}
        SIMULATED_FACTORY_PRICES[factory][date_str] = round(random.uniform(5800, 6200))

    # 河北工厂价格模拟 (假设在 6000 到 6400 元/吨之间波动)
    for factory in SIMULATED_FACTORIES["河北"]:
        if factory not in SIMULATED_FACTORY_PRICES:
            SIMULATED_FACTORY_PRICES[factory] = {}
        SIMULATED_FACTORY_PRICES[factory][date_str] = round(random.uniform(6000, 6400))

# --- Helper Function for Date Iteration ---
def daterange(start_date: date, end_date: date):
    """生成从 start_date 到 end_date (包含) 的日期对象"""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def parse_date(date_str: str) -> Optional[date]:
    """尝试将 YYYY-MM-DD 格式的字符串解析为 date 对象"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None

@mcp.tool()
def get_current_date() -> str:
    """返回今天的日期，格式为YYYY-MM-DD。"""
    return datetime.now().strftime("%Y-%m-%d")

@mcp.tool()
def get_auction_price(province: str, start_date: str, end_date: str) -> Dict[str, float]:
    """
    获取指定省份在指定日期范围内的原料气竞拍价格。
    输入参数：
        province: 省份名称 (例如 "内蒙古", "河北")
        start_date: 开始查询日期 (格式 "YYYY-MM-DD")
        end_date: 结束查询日期 (格式 "YYYY-MM-DD")
    返回：
        一个字典，键是日期字符串，值是对应的竞拍价格 (Dict[str, float])。
        如果某天找不到数据，则该日期不会包含在返回的字典中。
        如果日期格式错误或开始日期晚于结束日期，返回空字典。
    """
    print(f"模拟接口: 正在查询 {province} 省份从 {start_date} 到 {end_date} 的竞拍价...")
    start_date_obj = parse_date(start_date)
    end_date_obj = parse_date(end_date)

    if not start_date_obj or not end_date_obj or start_date_obj > end_date_obj:
        print(f"模拟接口: 日期格式错误或开始日期晚于结束日期。")
        return {}

    prices_over_time: Dict[str, float] = {}
    if province in SIMULATED_AUCTION_PRICES:
        province_data = SIMULATED_AUCTION_PRICES[province]
        for single_date_obj in daterange(start_date_obj, end_date_obj):
            date_str = single_date_obj.strftime("%Y-%m-%d")
            if date_str in province_data:
                prices_over_time[date_str] = province_data[date_str]
            # else:
            #     print(f"模拟接口: 未找到 {province} 省份在 {date_str} 的竞拍价数据。")

    print(f"模拟接口: 查询到竞拍价数据: {prices_over_time}")
    return prices_over_time

@mcp.tool()
def get_factory_prices(factory_names: List[str], start_date: str, end_date: str) -> Dict[str, Dict[str, float]]:
    """
    获取指定工厂列表在指定日期范围内的出厂价格。
    输入参数：
        factory_names: 工厂名称列表
        start_date: 开始查询日期 (格式 "YYYY-MM-DD")
        end_date: 结束查询日期 (格式 "YYYY-MM-DD")
    返回：
        一个嵌套字典，外层键是日期字符串，内层键是工厂名称，值是对应的价格 (Dict[str, Dict[str, float]])。
        如果某个工厂某天没有价格数据，则不会包含。如果某天所有指定工厂都没有价格，则该日期键不会存在。
        如果日期格式错误或开始日期晚于结束日期，返回空字典。
    """
    print(f"模拟接口: 正在查询工厂 {factory_names} 从 {start_date} 到 {end_date} 的出厂价格...")
    start_date_obj = parse_date(start_date)
    end_date_obj = parse_date(end_date)

    if not start_date_obj or not end_date_obj or start_date_obj > end_date_obj:
        print(f"模拟接口: 日期格式错误或开始日期晚于结束日期。")
        return {}

    prices_over_time: Dict[str, Dict[str, float]] = {}

    for single_date_obj in daterange(start_date_obj, end_date_obj):
        date_str = single_date_obj.strftime("%Y-%m-%d")
        daily_prices: Dict[str, float] = {}
        for factory in factory_names:
            if factory in SIMULATED_FACTORY_PRICES and date_str in SIMULATED_FACTORY_PRICES[factory]:
                daily_prices[factory] = SIMULATED_FACTORY_PRICES[factory][date_str]

        if daily_prices: # 仅当当天至少有一个工厂有价格时才记录
            prices_over_time[date_str] = daily_prices
        # else:
        #     print(f"模拟接口: 在 {date_str} 未找到任何指定工厂的价格数据。")

    print(f"模拟接口: 查询到价格数据: {prices_over_time}")
    return prices_over_time

#简单的示例
if __name__ == '__main__':
    # 模拟查询最近3天的内蒙古平均利润
    today_str = get_current_date()
    start_date_obj = parse_date(today_str) - timedelta(days=2)
    start_date_str = start_date_obj.strftime("%Y-%m-%d")
    end_date_str = today_str
