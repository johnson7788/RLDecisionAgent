"""MCP agent rollout implementation.

This module provides a rollout function for running MCP agents with scenarios.
Based on the art-e rollout.py structure.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass

import weave
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from servers.python.mcp_alphavantage.server_params import server_params

import art

from .checks import check_successful
from .utils import get_content_text

load_dotenv()

logging.getLogger("weave.trace.op").setLevel(logging.WARNING)


weave.init("mcp-agent-training")


@dataclass
class McpScenario:
    """A scenario for MCP agent evaluation."""

    task_description: str
    server_params: StdioServerParameters
    max_turns: int = 10


@weave.op()
async def rollout(
    model: art.Model,
    scenario: McpScenario,
    debug: bool = False,
) -> art.Trajectory:
    """Run an MCP agent rollout with server parameters."""
    debug = True
    print(f"[INFO] 开始 rollout，任务: {scenario.task_description}, 最大轮数: {scenario.max_turns}")

    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"task": scenario.task_description},
        metrics={
            "task_completed": False,
            "success": False,
            "ran_out_of_turns": False,
        },
        scenario=scenario,
    )

    system_prompt = (
        f"You are an MCP (Model Context Protocol) agent.\n"
        f"You have access to MCP tools through the server. Use them to complete your task.\n"
        f"You have a total of {scenario.max_turns} turns. Only use tool calls.\n"
        f"Call 'complete_task' when finished."
    )

    try:
        print(f"[INFO] 连接 MCP 服务器...")
        async with stdio_client(scenario.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print(f"[INFO] MCP 服务器连接成功，初始化 session")
                await session.initialize()
                tools_result = await session.list_tools()
                print(f"[INFO] MCP server 返回 {len(tools_result.tools)} 个工具")

                tool_schemas = []
                for tool in tools_result.tools:
                    tool_schema = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or f"MCP tool: {tool.name}",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        },
                    }
                    tool_schemas.append(tool_schema)

                if debug:
                    print(f"[DEBUG] Available MCP tools: {[t['function']['name'] for t in tool_schemas]}")

                # Add completion tool
                tool_schemas.append({
                    "type": "function",
                    "function": {
                        "name": "complete_task",
                        "description": "Complete the task with a summary",
                        "parameters": {
                            "type": "object",
                            "properties": {"summary": {"type": "string", "description": "Summary of accomplishments"}},
                            "required": ["summary"],
                        },
                    },
                })
                traj.tools = tool_schemas

                traj.messages_and_choices = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please complete this task: {scenario.task_description}"},
                ]

                if debug:
                    print(f"[DEBUG] 初始消息: {traj.messages()}")

                num_turns = 0
                task_completed = False

                while num_turns < scenario.max_turns and not task_completed:
                    num_turns += 1
                    print(f"[INFO] 开始第 {num_turns} 轮交互")

                    try:
                        async with traj.track_duration("llm_completion"):
                            client = AsyncOpenAI(
                                api_key=model.inference_api_key,
                                base_url=model.inference_base_url,
                            )
                            response = await client.chat.completions.create(
                                model=model.inference_model_name or model.name,
                                messages=traj.messages(),
                                temperature=1.0,
                                tools=tool_schemas,
                                max_completion_tokens=8000,
                            )

                        choice = response.choices[0]
                        traj.messages_and_choices.append(choice)
                        if debug:
                            print(f"[DEBUG] LLM Choice: {choice.message}")

                        # Handle tool calls
                        if choice.message.tool_calls:
                            for tool_call in choice.message.tool_calls:
                                try:
                                    tool_args = json.loads(tool_call.function.arguments)
                                    print(f"[INFO] 执行工具调用: {tool_call.function.name}, args={tool_args}")

                                    if tool_call.function.name == "complete_task":
                                        traj.metrics["task_completed"] = True
                                        print(f"[INFO] 尝试完成任务, summary={tool_args['summary']}")
                                        try:
                                            traj.metrics["success"] = await check_successful(traj)
                                            print(f"[INFO] 任务完成状态: {traj.metrics['success']}")
                                        except Exception as e:
                                            print(f"[ERROR] 检查任务成功状态失败: {e}")
                                        task_completed = True
                                    else:
                                        result = await session.call_tool(tool_call.function.name, tool_args)
                                        content_text = get_content_text(result)
                                        print(f"[INFO] 工具 {tool_call.function.name} 调用返回长度: {len(content_text)}")
                                        if len(content_text) > 20000:
                                            print(f"[WARNING] 工具返回内容过长, 前后各1000字符输出")
                                            print(content_text[:1000])
                                            print(content_text[-1000:])
                                            raise Exception("Tool call result too long")

                                        traj.messages_and_choices.append(
                                            {"role": "tool", "tool_call_id": tool_call.id, "content": content_text}
                                        )
                                        if debug:
                                            print(f"[DEBUG] 工具调用结果: {content_text[:200]}...")

                                except Exception as e:
                                    print(f"[ERROR] 工具调用出错: {e}")
                                    traj.messages_and_choices.append(
                                        {"role": "tool", "tool_call_id": tool_call.id, "content": f"Error: {str(e)}"}
                                    )
                        else:
                            print(f"[INFO] 本轮无工具调用, 继续下一轮或结束")
                            break

                    except Exception as e:
                        print(f"[ERROR] 第 {num_turns} 轮交互出错: {e}")
                        break

    except Exception as e:
        print(f"[ERROR] MCP server 错误: {e}")

    if not task_completed and num_turns == scenario.max_turns:
        traj.metrics["ran_out_of_turns"] = True
        print(f"[INFO] 已达到最大轮数 {num_turns}，任务未完成")

    traj.metrics["num_turns"] = num_turns
    print(f"[INFO] rollout 结束, 总轮数: {num_turns}, 完成状态: {traj.metrics['task_completed']}")

    if debug:
        print(f"[DEBUG] 最终消息列表:")
        for message in traj.messages_and_choices:
            print(message)

    return traj.finish()


async def test_rollout():
    model = art.Model(
        name="o4-mini",
        project="mcp-agent-training",
        inference_model_name="deepseek-chat",
        inference_api_key=os.getenv("DEEPSEEK_API_KEY"),
        inference_base_url="https://api.deepseek.com/v1",
    )

    scenario = McpScenario(
        task_description="Use search_symbol to find emerging biotech companies by searching with keywords 'biotech' and review the results.",
        server_params=server_params,
    )

    traj = await rollout(model, scenario, debug=True)
    print(traj)


async def main():
    """Run test scenario."""
    print("=== Testing Python MCP AlphaVantage Server ===")
    await test_rollout()


if __name__ == "__main__":
    asyncio.run(main())
