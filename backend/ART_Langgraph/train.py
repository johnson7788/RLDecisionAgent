#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:30
# @File  : train.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 

# -*- coding: utf-8 -*-
"""
训练一个使用 LangGraph Tools 的 ReAct Agent（基于 ART 的 GRPO 强化训练）
- 工具：的网页搜索 + “提交最终答案”工具
- 训练：迭代采样 -> 组内相对打分(RULER, 可选) -> model.train()
依赖：
  uv pip install -U "openpipe-art[backend,langgraph]>=0.4.9" langchain-core pydantic tenacity litellm
"""

import os
import uuid
import asyncio
from textwrap import dedent
from typing import List, Optional

import art
from art.langgraph import init_chat_model, wrap_rollout
from art.utils import iterate_dataset
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt
from litellm import acompletion
from zai import ZhipuAiClient

# ---------- 配置 ----------
# 任选一个可训练且支持 tools 的基础模型（Qwen2.5 系列在文档中常被用作示例）
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "web-search-agent-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"

# RULER 评估模型（可选；需相应 API Key）
RULER_MODEL = os.getenv("RULER_MODEL", "openai/o4-mini")

WebSearchClient = ZhipuAiClient(api_key=os.environ["ZHIPU_API_KEY"])


# ---------- 数据结构 ----------
class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str

class FinalAnswer(BaseModel):
    answer: str
    source_urls: List[str]

class Scenario(BaseModel):
    id: str
    question: str
    answer: str

class WebSearchScenario(BaseModel):
    step: int
    scenario: Scenario

class ProjectTrajectory(art.Trajectory):
    final_answer: Optional[FinalAnswer] = None

async def search_web(keyword: str) -> List[WebSearchResult]:
    """
    真实的网络搜索函数
    """
    response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=15,  # 返回结果的条数，范围1-50，默认10
        search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
        content_size="high"  # 控制网页摘要的字数，默认medium
    )
    return [
        WebSearchResult(
            url=item['url'],
            title=item['title'],
            snippet=item['content']
        ) for item in response['search_result']
    ]

# ---------- 可选：正确性评估（仅做指标记录，不参与训练权重更新）----------

class CorrectnessJudgeResponse(BaseModel):
    reasoning: str = Field(description="why")
    accept: bool = Field(description="accept or not")

@retry(stop=stop_after_attempt(3))
async def judge_correctness(s: Scenario, answer: str) -> CorrectnessJudgeResponse:
    system_prompt = "Judge whether the AI answer matches the reference."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Q: {s.question}\nRef: {s.answer}\nAI: {answer}"},
    ]
    resp = await acompletion(
        model="openai/gpt-4o-mini",
        messages=messages,
        response_format=CorrectnessJudgeResponse,
    )
    return CorrectnessJudgeResponse.model_validate_json(
        resp.choices[0].message.content or "{}"
    )

# ---------- rollout：LangGraph + Tools ----------
async def rollout(model: art.Model, web_search_scenario: WebSearchScenario) -> ProjectTrajectory:
    scenario = web_search_scenario.scenario
    MAX_TURNS = 5

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": scenario.id, "step": web_search_scenario.step},
    )

    system_prompt = dedent(f"""
    You are a web search agent. Use tools to search the web.
    When done, call return_final_answer_tool(answer, source_urls).
    """ )

    final_answer: Optional[FinalAnswer] = None

    @tool
    async def web_search_tool(query: str) -> List[dict]:
        """Search the web by a query and return a list of results."""
        results = await search_web(query)
        return [result.model_dump() for result in results]

    @tool
    def return_final_answer_tool(answer: str, source_urls: List[str]) -> dict:
        """Return final answer with source URLs."""
        nonlocal final_answer
        final_answer = FinalAnswer(answer=answer, source_urls=source_urls)
        return final_answer.model_dump()

    tools = [web_search_tool, return_final_answer_tool]

    # 关键：用 ART 的 init_chat_model 注入可训练/可记录的聊天模型
    chat_model = init_chat_model(MODEL_NAME, temperature=1.0)
    agent = create_react_agent(chat_model, tools)

    await agent.ainvoke(
        {"messages": [SystemMessage(content=system_prompt),
                      HumanMessage(content=scenario.question)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": MAX_TURNS},
    )

    if final_answer:
        traj.final_answer = final_answer
        try:
            judge = await judge_correctness(scenario, final_answer.answer)
            traj.metrics["correct"] = float(judge.accept)
        except Exception:
            pass
    return traj

# ---------- 训练主程序 ----------
async def main():
    # 选择后端：本地 GPU 或远端 SkyPilot（无本地 GPU 时）
    if USE_LOCAL_BACKEND:
        from art.local.backend import LocalBackend
        backend = LocalBackend()
    else:
        from art.skypilot.backend import SkyPilotBackend
        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=os.getenv("ART_SKYPILOT_CLUSTER", "art-cluster"),
            gpu=os.getenv("ART_GPU", "A100"),
        )

    # 声明/注册模型（名称/工程名便于组织与复用）
    model = art.Model(name=MODEL_NAME, project=PROJECT_NAME)
    await model.register(backend)

    # 构造小型训练集（替换成你的真实场景）
    training_scenarios = [
        Scenario(
            id="1",
            question="Who is the CEO of OpenAI?",
            answer="Sam Altman",
        ),
        Scenario(
            id="2",
            question="What is the capital of France?",
            answer="Paris",
        ),
    ]

    # 训练参数
    training_config = {
        "groups_per_step": 2,
        "num_epochs": 2,
        "rollouts_per_group": 4,
        "learning_rate": 1e-5,
        "max_steps": 5,
    }

    # 训练数据迭代器
    it = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    # 是否使用 RULER（无 judge key 时自动回退为“直接训练已完成的轨迹”）
    try:
        from art.rewards import ruler_score_group
        use_ruler = True
    except Exception:
        use_ruler = False

    for batch in it:
        print(f"[train] step={batch.step} epoch={batch.epoch}")

        # 组装 TrajectoryGroup：每个样本 rollout 多条轨迹
        groups = []
        for s in batch.items:
            groups.append(
                art.TrajectoryGroup(
                    wrap_rollout(model, rollout)(
                        model, WebSearchScenario(step=batch.step, scenario=s)
                    )
                    for _ in range(training_config["rollouts_per_group"])
                )
            )

        # 收集轨迹（ART 会自动串联推理与训练，后端保存并加载最新 LoRA）
        finished = await art.gather_trajectory_groups(
            groups, pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        # 打分 & 训练
        if use_ruler:
            judged = []
            for g in finished:
                jg = await ruler_score_group(g, RULER_MODEL, debug=True)
                judged.append(jg)
            await model.train(
                judged,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
                _config={"logprob_calculation_chunk_size": 8},
            )
        else:
            # 没有 RULER 的情况下，也可以直接训练（若你在 rollout 中自己设置了 reward）
            await model.train(
                finished,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            )

        if batch.step >= training_config["max_steps"]:
            break

if __name__ == "__main__":
    asyncio.run(main())
