import logging
import os
import sys
import click
import httpx
import uvicorn
from uvicorn import Config, Server
import asyncio
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv
from agent_executor import KnowledgeAgentExecutor
from agent import KnowledgeAgent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def start_server(host, port, mcp_config, select_tool_names):
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id='knowledge_search',
        name='知识库问答能力',
        description='可以从文档、个人知识库和指南中检索信息并回答用户问题',
        tags=['知识检索', '多轮问答', '智能助手'],
        examples=['帕金森的治疗方案有哪些？', '我在指南中能找到哪些营养建议？'],
    )
    CARD_URL = os.environ.get('CARD_URL', f'http://{host}:{port}/')
    agent_card = AgentCard(
        name='知识库问答 Agent',
        description='可以根据用户的问题从多个知识库中检索并回答',
        url=CARD_URL,
        version='1.0.0',
        defaultInputModes=KnowledgeAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=KnowledgeAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )
    # 启动服务
    agent_executor = KnowledgeAgentExecutor(mcp_config, select_tool_names)
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

    config = Config(app=server.build(), host=host, port=port, log_level="info")
    server_instance = Server(config)
    await server_instance.serve()

@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10000)
@click.option('--mcp', default="mcp_config.json", help='MCP 配置文件路径')
@click.option('--select_tool_names', default="search_document_db,search_personal_db,search_guideline_db", help='使用的内部工具，逗号分隔')
def main(host, port, mcp, select_tool_names):
    select_tool_names = select_tool_names.split(",")
    asyncio.run(start_server(host, port, mcp, select_tool_names))

if __name__ == '__main__':
    main()