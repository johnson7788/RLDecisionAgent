#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/16 12:43
# @File  : cli_model_load.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import asyncio
import os
import time

# 将超时设置为一个很高的值（例如10分钟），以便进行测试
os.environ["ART_SERVER_TIMEOUT"] = "600"

from art.backend import Backend
from art.model import TrainableModel

# -----------------------------------------------------------------
# 你需要从你的训练脚本（例如 mcp_rl/train.py）中找到这个值
# -----------------------------------------------------------------
# TrainableModel现在需要一个'name'（你正在训练的模型的名字）
# 和一个'base_model'（你要微调的模型的名字）。
MODEL_NAME = "my-qwen-test"  # <--- 给你的模型起一个名字
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # <--- 这是基础模型
PROJECT_NAME = "test-loading-project"


async def main():
    # 这个客户端会连接到你在第1步中启动的服务器
    backend = Backend()

    # 定义我们希望后端加载的模型
    model = TrainableModel(name=MODEL_NAME, base_model=BASE_MODEL, project=PROJECT_NAME)

    print(f"正在请求后端服务器加载模型: {model.name} (base: {model.base_model})")
    print("这可能需要一些时间...")
    start_time = time.time()

    try:
        # 这个调用会通过网络请求，让服务器开始加载模型
        base_url, api_key = await backend._prepare_backend_for_training(model, config=None)

        end_time = time.time()
        print("\n✅ 后端成功加载模型！")
        print(f"   耗时: {end_time - start_time:.2f} 秒")
        print(f"   模型现在被托管在: {base_url}")

    except Exception as e:
        end_time = time.time()
        print(f"\n❌ 加载模型时发生错误 (耗时: {end_time - start_time:.2f} 秒)")
        print(f"   错误详情: {e}")


if __name__ == "__main__":
    asyncio.run(main())
