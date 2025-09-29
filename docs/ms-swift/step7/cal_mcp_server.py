#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/29
# @File  : calculator_service.py
# @Author: your_name
# @Desc  : 计算器服务实现

from fastmcp import FastMCP
from typing import List, Dict, Optional

# ========== FastMCP 服务 ==========
mcp = FastMCP("简单计算器服务")

# ========== 计算工具 ==========
@mcp.tool()
def add(a: float, b: float) -> float:
    """加法操作"""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """减法操作"""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """乘法操作"""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> Optional[float]:
    """除法操作，防止除以零"""
    if b == 0:
        return None  # 除以零返回 None
    return a / b

@mcp.tool()
def power(a: float, b: float) -> float:
    """幂运算"""
    return a ** b

@mcp.tool()
def sqrt(a: float) -> Optional[float]:
    """平方根运算，返回负数输入为 None"""
    if a < 0:
        return None  # 返回 None 表示无效输入
    return a ** 0.5

# ========== 运行服务 ==========
if __name__ == '__main__':
    mcp.run(transport="sse", host="127.0.0.1", port=9000)
