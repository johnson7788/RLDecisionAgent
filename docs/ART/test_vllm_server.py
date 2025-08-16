import asyncio
import os
from typing import Any

from openai import AsyncOpenAI
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from art.vllm.server import openai_server_task


async def main():
    """
    1. 正在初始化 vLLM 引擎。
    2. 正在启动 OpenAI 兼容的服务器。
    3. 正在向服务器发送请求。
    4. 正在关闭服务器。
    """
    # 1. Initialize the vLLM engine
    #初始化1个vllm模型，然后使用下面的AsyncLLMEngine调用
    engine_args = AsyncEngineArgs(model="Qwen/Qwen2.5-0.5B-Instruct")
    engine = await AsyncLLMEngine.from_engine_args(engine_args)

    # 2. Start the OpenAI-compatible server
    server_config: dict[str, Any] = {
        "engine_args": {
            "model": "my-qwen-test",
        },
        "server_args": {
            "port": 8000,
            "host": "localhost",
        },
        "log_file": "vllm.log",
    }

    # The engine needs to be passed to the server task
    # The server will be started as a background task
    server_task = await openai_server_task(engine, server_config)

    # 3. Send a request to the server
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    try:
        print("Testing /v1/models endpoint...")
        models = await client.models.list()
        print(f"Success! Models: {[m.id for m in models.data]}")

        print("\nTesting /v1/completions endpoint...")
        completion = await client.completions.create(
            model="gpt2",
            prompt="Hello, my name is",
            max_tokens=5,
            temperature=0.0,
        )
        print(f"Success! Completion: {completion.choices[0].text}")

    finally:
        # 4. Shut down the server
        print("\nShutting down server...")
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            print("Server successfully shut down.")

    # Stop the engine
    await engine.abort()


if __name__ == "__main__":
    # Set dummy OpenAI API key for the client
    os.environ["OPENAI_API_KEY"] = "dummy"
    asyncio.run(main())
