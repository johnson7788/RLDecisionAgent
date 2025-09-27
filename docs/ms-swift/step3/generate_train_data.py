import argparse
import logging
from typing import Any, Dict, List
from uuid import uuid4
import json
import os

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendStreamingMessageRequest,
)
from mcp_client import get_mcp_tools


PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"
EXTENDED_AGENT_CARD_PATH = "/agent/authenticatedExtendedCard"


async def get_agent_response(client: A2AClient, question: str) -> List[Dict[str, Any]]:
    """Sends a question to the agent and returns the conversation history in the desired format."""
    conversations = [{"from": "human", "value": question}]

    parts = [{"kind": "text", "text": question}]
    multiturn_first: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": parts,
            "messageId": uuid4().hex,
            "metadata": {"language": "English", "user_id": 123456},
        },
    }
    streaming_request = SendStreamingMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**multiturn_first)
    )
    stream_response = client.send_message_streaming(streaming_request)

    text_response = ""
    async for chunk in stream_response:
        chunk_data = chunk.model_dump(mode="json", exclude_none=True)

        result = chunk_data.get("result")
        if not result or result.get("kind") != "status-update":
            continue

        status = result.get("status")
        if not status:
            continue

        message = status.get("message")
        if not message:
            continue

        parts = message.get("parts")
        if not parts:
            continue

        for part in parts:
            if part.get("kind") == "data":
                data1 = part.get("data")
                if data1 and data1.get("data"):
                    for item in data1.get("data"):
                        # This is a tool call
                        if item and item.get("type") == "function":
                            function_call = item.get("function")
                            if function_call:
                                conversations.append(
                                    {
                                        "from": "function_call",
                                        "value": json.dumps(function_call, ensure_ascii=False),
                                    }
                                )
                        # This is a tool observation
                        if item and item.get("tool_output") is not None:
                            tool_output = item.get("tool_output")
                            observation_value = ""
                            try:
                                parsed_output = json.loads(tool_output)
                                if isinstance(parsed_output, (dict, list)):
                                    observation_value = tool_output
                                else:
                                    observation_value = json.dumps({"result": parsed_output})
                            except (json.JSONDecodeError, TypeError):
                                observation_value = json.dumps({"result": tool_output})

                            conversations.append({"from": "observation", "value": observation_value})
            elif part.get("kind") == "text":
                text = part.get("text")
                if text:
                    text_response += text

    if text_response:
        conversations.append({"from": "gpt", "value": text_response.strip()})

    return conversations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从问题文件逐条提问 Agent，收集对话与工具信息，写入 JSONL。"
    )
    parser.add_argument(
        "-q",
        "--questions-file",
        default="./questions.txt",
        help="输入问题文件路径（每行一个问题）。默认：./questions.txt",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="train.jsonl",
        help="输出 JSONL 文件路径。默认：train.jsonl",
    )
    parser.add_argument(
        "--mcp-config",
        default="./mcp_config.json",
        help="MCP 配置文件路径。默认：./mcp_config.json",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("A2A_BASE_URL", "http://localhost:10066"),
        help="Agent 服务 Base URL。可用环境变量 A2A_BASE_URL 覆盖。默认：http://localhost:10066",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 读取 MCP 配置
    try:
        with open(args.mcp_config, "r", encoding="utf-8") as f:
            mcp_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"MCP 配置文件不存在，请检查 {args.mcp_config}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"MCP 配置文件不是合法 JSON：{e}")
        return

    mcp_servers = mcp_config.get("mcpServers", {})
    all_tools = []
    for server_name, server_info in mcp_servers.items():
        if not server_info.get("disabled"):
            url = server_info.get("url")
            if url:
                logger.info(f"Fetching tools from '{server_name}' at {url}...")
                server_tools = await get_mcp_tools(url)
                all_tools.extend(server_tools)

    logger.info(f"发现 {len(all_tools)} 个工具通过读取 MCP 的 Server 端")

    # 校验输入问题文件
    if not os.path.exists(args.questions_file):
        logger.error(f"问题文件不存在：{args.questions_file}")
        return

    async with httpx.AsyncClient(timeout=60) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=args.base_url)
        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(f"尝试获取 Agent 配置信息: {args.base_url}{PUBLIC_AGENT_CARD_PATH}")
            _public_card = await resolver.get_agent_card()
            final_agent_card_to_use = _public_card

            if _public_card.supportsAuthenticatedExtendedCard:
                try:
                    auth_headers_dict = {"Authorization": "Bearer dummy-token-for-extended-card"}
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={"headers": auth_headers_dict},
                    )
                    final_agent_card_to_use = _extended_card
                except Exception as e_extended:
                    logger.warning(f"Failed to get extended card: {e_extended}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to get AgentCard: {e}", exc_info=True)
            raise RuntimeError("Cannot get agent card, cannot continue.") from e

        client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)
        logger.info("A2AClient 初始化完成.")

        # 读取问题并生成训练数据
        with open(args.questions_file, "r", encoding="utf-8") as f_in, open(
            args.output_file, "w", encoding="utf-8"
        ) as f_out:
            questions = [line.strip() for line in f_in if line.strip()]
            if not questions:
                logger.warning("问题文件为空，没有可处理的问题。")
            for question in questions:
                logger.info(f"Processing question: {question}")
                conversations = await get_agent_response(client, question)
                training_entry = {"conversations": conversations, "tools": all_tools}
                f_out.write(json.dumps(training_entry, ensure_ascii=False) + "\n")
                logger.info(f"Finished processing and saved entry for question: {question}")

        logger.info(f"Successfully generated training data and saved to {args.output_file}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
