#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:41
# @File  : model_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

# -*- coding: utf-8 -*-
"""
用训练过的同名模型跑一次 ReAct 推理测试（同样走 LangGraph tools）
注意：
- 若你在同一后端进程中连续训练->测试，后端已加载最近的 LoRA。
- 若在新进程测试，请保持相同的 model.name / project，并连接到相同后端（本地或 SkyPilot）。
"""

import os
import uuid
import asyncio
from dataclasses import asdict
from textwrap import dedent
from typing import List, Optional

import art
from art.langgraph import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel

# ---------- 与训练保持一致 ----------
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-7B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "email-agent-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"

# ---------- 业务工具（与训练相同或真实实现）----------
class EmailResult(BaseModel):
    message_id: str
    subject: str
    from_address: str
    date: str
    snippet: str

def search_emails(inbox: str, keywords: List[str], sent_before: str) -> List[EmailResult]:
    return [EmailResult(
        message_id="msg_999",
        subject=f"[TEST] {keywords[0]} found",
        from_address="sender@example.com",
        date="2024-01-15",
        snippet=f"[TEST] body with {keywords[0]}",
    )]

def read_email(message_id: str) -> Optional[EmailResult]:
    return EmailResult(
        message_id=message_id,
        subject="[TEST] full subject",
        from_address="sender@example.com",
        date="2024-01-15",
        snippet="[TEST] full body...",
    )

async def main():
    # 连接与注册后端
    if USE_LOCAL_BACKEND:
        from art.local.backend import LocalBackend
        backend = LocalBackend()
    else:
        from art.skypilot.backend import SkyPilotBackend
        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=os.getenv("ART_SKYPILOT_CLUSTER", "art-cluster"),
            gpu=os.getenv("ART_GPU", "A100"),
        )

    model = art.Model(name=MODEL_NAME, project=PROJECT_NAME)
    await model.register(backend)

    system_prompt = dedent("""
    You are an email search agent. Use tools to find and cite evidence.
    When done, provide a concise answer.
    """)
    inbox = "user@company.com"
    today = "2024-01-20"

    @tool
    def search_inbox_tool(keywords: List[str]) -> List[dict]:
        results = search_emails(inbox, keywords, today)
        return [asdict(r) for r in results]

    @tool
    def read_email_tool(message_id: str) -> Optional[dict]:
        email = read_email(message_id)
        return email.model_dump() if email else None

    tools = [search_inbox_tool, read_email_tool]

    # 用 ART 的 init_chat_model 获取可用的聊天模型（后端会加载最近训练好的 LoRA）
    chat_model = init_chat_model(MODEL_NAME, temperature=0.7)
    agent = create_react_agent(chat_model, tools)

    res = await agent.ainvoke(
        {"messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="查找包含 budget 的邮件，并告诉我最关键的一条结论。"),
        ]},
        config={"configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": 10},
    )
    print("[TEST] agent finished. See backend logs / tracing for details.")

if __name__ == "__main__":
    asyncio.run(main())
