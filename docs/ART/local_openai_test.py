#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/22 20:56
# @File  : local_openai_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试本地的llm代理是否OK
import os
import dotenv
import asyncio, litellm
dotenv.load_dotenv()
litellm._turn_on_debug()

def streaming_main():
    # openai call
    response = litellm.completion(
        model="o4-mini",
        messages=[{"role":"user","content":"你好"}],
        base_url="http://localhost:6688",
        api_key=os.environ["OPENAI_API_KEY"],
        stream=True
    )

    for chunk in response:
        print(chunk)

async def main():
    resp = await litellm.acompletion(
        model="o4-mini",
        messages=[{"role":"user","content":"Hello"}],
        base_url="http://localhost:6688",
        api_key=os.environ["OPENAI_API_KEY"]
    )

    # 正常情况下，LiteLLM 返回 dict（已序列化的对象）
    print(type(resp))
    print(resp)
    print(resp.model_dump())

if __name__ == '__main__':
    streaming_main()
    asyncio.run(main())
