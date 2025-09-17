#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27
# @File  : train_query_agent.py
# @Author: johnson
# @Desc  : 训练一个“查询”ReAct Agent（ART + LangGraph，GRPO），从 questions.txt 读取问题

import logging
logging.basicConfig(level=logging.DEBUG)
import os
import json
import uuid
import time
import asyncio
import dotenv
import wandb
from dataclasses import dataclass
from textwrap import dedent
from typing import List, Dict, Any, Optional

import art
from art.langgraph import init_chat_model, wrap_rollout
from art.utils import iterate_dataset
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from reward import search_reward, format_reward

# ==== 你的 Web 搜索客户端（示例：ZhipuAiClient）====
from zai import ZhipuAiClient
dotenv.load_dotenv()
WebSearchClient = ZhipuAiClient(api_key=os.environ["ZHIPU_API_KEY"])

# ---------------- wandb: 运行配置 ----------------
NAME = os.getenv("ART_NAME", "query-web")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "content-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"

WANDB_PROJECT = os.getenv("WANDB_PROJECT", PROJECT_NAME)
WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # 可空
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}")
print(f"{NAME} - {MODEL_NAME} - {PROJECT_NAME} - {os.getenv('WANDB_BASE_URL', '')}")

# ----------------- 数据结构 -----------------
class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str

class FinalQAResult(BaseModel):
    task: List[Dict[str, Any]]                 # 单元素任务数组：[{"type":"qa","data":{"question":..., "text": 答案}}]
    sources: List[str] = Field(default_factory=list)  # [1],[2]... 对应的 URL 列表

@dataclass
class QueryScenario:
    id: str
    prompt: str
    input_task: List[Dict[str, Any]]  # [{"type":"qa","data":{"question": "...", "text": ""}}]

class ProjectTrajectory(art.Trajectory):
    final: Optional[FinalQAResult] = None

# ----------------- 从文件加载问题 -----------------
def _default_questions_path() -> str:
    # 以当前文件所在目录为基准，默认在上一级目录的 questions.txt
    here = os.path.dirname(os.path.abspath(__file__))
    question_path = os.path.abspath(os.path.join(here, "..", "questions.txt"))
    return question_path

def load_questions(path: Optional[str] = None) -> List[str]:
    qpath = path or os.getenv("QUESTIONS_PATH") or _default_questions_path()
    print(f"开始加载训练问题：{qpath}")
    if not os.path.exists(qpath):
        raise FileNotFoundError(f"questions.txt 未找到：{qpath}")
    questions: List[str] = []
    with open(qpath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 跳过注释行（可选）
            if s.startswith("#"):
                continue
            questions.append(s)
    if not questions:
        raise ValueError("questions.txt 为空或没有有效问题行。")
    return questions

# ----------------- 工具：web 搜索 -----------------
async def search_web(keyword: str) -> List[WebSearchResult]:
    response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=4,
        search_recency_filter="noLimit",
        content_size="high"
    )
    if not response.search_result:
        return []
    return [
        WebSearchResult(url=sr.link, title=sr.title, snippet=sr.content)
        for sr in response.search_result
    ]

# ----------------- rollout（核心）：LangGraph + Tools -----------------
async def rollout(model: art.Model, scenario: QueryScenario) -> ProjectTrajectory:
    MAX_TURNS = 16
    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": scenario.id}
    )

    # ====== 提示词：查询 Agent 要求 ======
    system_prompt = dedent("""
    你是一个数据查询与分析助手（Query Agent）。你的任务：
    1) 读取用户提供的单元素 JSON 任务数组（task），其中包含一个 {type:"qa", data:{question, text}}；
    2) 直接回答 data.question，必要时使用 web_search_tool 构造精准查询（包含主题、地域、时间、行业/政策关键词），多源交叉验证；
    3) 输出 2~6 句，包含可核验事实（数字/日期/机构/地名等），并在句尾添加来源引用 [n]（n 从 1 开始，对应 sources 中 URL 的顺序）；
    4) 不得修改输入 JSON 的结构和字段，只能用最终答案覆盖 data.text；
    5) 完成后调用 return_final_answer_tool(task, sources)：
       - task：保持与输入一致的数组，仅 data.text 被替换为你的答案（包含 [n] 引用）；
       - sources：去重后的 URL 列表，顺序与 [n] 对应；
    6) 不要在普通对话中粘贴 JSON，务必用工具返回最终 JSON。
    """)

    # ====== 工具定义 ======
    @tool
    async def web_search_tool(query: str) -> List[dict]:
        """通过关键词查询网页，返回[{url,title,snippet}, ...]。"""
        results = await search_web(query)
        return [r.model_dump() for r in results]

    final: Optional[FinalQAResult] = None

    @tool
    def return_final_answer_tool(task: List[Dict[str, Any]], sources: List[str]) -> dict:
        """返回最终 JSON：保持原格式的 task 与 sources。"""
        nonlocal final
        try:
            final = FinalQAResult(task=task, sources=sources or [])
            return final.model_dump()
        except ValidationError as e:
            # 让 Agent 收到错误并自我修复
            return {"error": f"ValidationError: {str(e)}"}

    tools = [web_search_tool, return_final_answer_tool]

    chat_model = init_chat_model(MODEL_NAME, temperature=0.8)
    agent = create_react_agent(chat_model, tools)

    # ====== 执行 Agent ======
    user_msg = dedent(f"""
    请严格按系统要求处理以下 JSON 查询任务（只替换 data.text）：
    {json.dumps(scenario.input_task, ensure_ascii=False)}
    """)

    await agent.ainvoke(
        {"messages": [SystemMessage(content=system_prompt),
                      HumanMessage(content=user_msg)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": MAX_TURNS},
    )

    # ====== 计算奖励 ======
    if final:
        traj.final = final

        # 收集工具阶段见过的 URL，用于一致性校验
        tool_urls_seen: List[str] = []
        try:
            for m in traj.messages_and_choices:
                if m.get("role") == "tool" and m.get("name") == "web_search_tool":
                    content = m.get("content")
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except Exception:
                            content = None
                    if isinstance(content, list):
                        for rec in content:
                            u = (rec or {}).get("url")
                            if isinstance(u, str):
                                tool_urls_seen.append(u)
        except Exception:
            pass

        # 仍然沿用原奖励：格式 + 搜索一致性
        fr = 0.0
        try:
            fr = format_reward(scenario.input_task, final.task)
        except Exception:
            # 若 format_reward 期望更复杂大纲结构，失败时给最低权重
            fr = 0.0

        sr = 0.0
        try:
            sr = search_reward(final.task, final.sources, tool_urls_seen=tool_urls_seen)
        except Exception:
            sr = 0.0

        traj.reward = 0.5 * fr + 0.5 * sr
        traj.metrics["format_reward"] = fr
        traj.metrics["search_reward"] = sr
        traj.metrics["sources_count"] = len(set(final.sources))

    return traj

# ----------------- wandb 记录 -----------------
def _log_batch_to_wandb(*, batch, finished_groups):
    trajectories = []
    for g in finished_groups:
        if hasattr(g, "trajectories"):
            trajectories.extend(getattr(g, "trajectories"))
        else:
            try:
                trajectories.extend(list(g))
            except Exception:
                pass

    table = wandb.Table(columns=["scenario_id", "format_reward", "search_reward", "total_reward", "sources"])
    for t in trajectories[:50]:
        sid = (getattr(t, "metadata", {}) or {}).get("scenario_id", "")
        fr = (getattr(t, "metrics", {}) or {}).get("format_reward", 0.0)
        sr = (getattr(t, "metrics", {}) or {}).get("search_reward", 0.0)
        rw = getattr(t, "reward", 0.0)
        srcs = ", ".join(getattr(getattr(t, "final", None), "sources", []) or [])
        table.add_data(sid, fr, sr, rw, srcs)

    wandb.log({
        "train/step": batch.step,
        "train/epoch": batch.epoch,
        "samples/trajectories": table
    }, step=batch.step)

# ----------------- 构造训练集 -----------------
def build_scenarios_from_questions(questions: List[str]) -> List[QueryScenario]:
    """
    把每一条问题行转换为单元素 task：
    [{"type": "qa", "data": {"question": <q>, "text": ""}}]
    """
    scenarios: List[QueryScenario] = []
    for i, q in enumerate(questions, start=1):
        task = [{"type": "qa", "data": {"question": q, "text": ""}}]
        sid = f"q_{i}"
        prompt = "回答 data.question（仅填写 data.text，加入 [n] 引用并返回 sources）。"
        scenarios.append(QueryScenario(id=sid, prompt=prompt, input_task=task))
    return scenarios

# ----------------- 主训练循环 -----------------
async def main():
    # wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY if WANDB_ENTITY else None,
        name=WANDB_RUN_NAME,
        config={
            "art_project": PROJECT_NAME,
            "art_name": NAME,
            "base_model": MODEL_NAME,
            "backend": "local" if USE_LOCAL_BACKEND else "skypilot",
        },
        settings=wandb.Settings(start_method="thread"),
    )
    wandb.define_metric("*", step_metric="train/step")

    # Backend
    if USE_LOCAL_BACKEND:
        from art.local.backend import LocalBackend
        backend = LocalBackend()
    else:
        from art.skypilot.backend import SkyPilotBackend
        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=os.getenv("ART_SKYPILOT_CLUSTER", "art-cluster"),
            gpu=os.getenv("ART_GPU", "A100"),
        )

    model = art.TrainableModel(name=NAME, project=PROJECT_NAME, base_model=MODEL_NAME)
    await model.register(backend)

    # 从 questions.txt 加载问题并构造场景
    questions = load_questions()
    scenarios = build_scenarios_from_questions(questions)

    training_config = {
        "groups_per_step": 2,
        "num_epochs": int(os.environ.get("NUM_EPOCHS", "2")),
        "rollouts_per_group": 3,
        "learning_rate": 1e-5,
        "max_steps": 6,
    }
    wandb.config.update(training_config)

    # 数据迭代器
    it = iterate_dataset(
        scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    for batch in it:
        print(f"[train] step={batch.step} epoch={batch.epoch}")

        # 组装 TrajectoryGroup：每个样本 rollout 多条轨迹
        groups = []
        for s in batch.items:
            groups.append(
                art.TrajectoryGroup(
                    wrap_rollout(model, rollout)(model, s)
                    for _ in range(training_config["rollouts_per_group"])
                )
            )

        # 收集轨迹
        finished = await art.gather_trajectory_groups(
            groups, pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        _log_batch_to_wandb(batch=batch, finished_groups=finished)

        # 用我们在 rollout 里写入的 reward 做 GRPO
        await model.train(
            finished,
            config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
        )

        if batch.step >= training_config["max_steps"]:
            break

    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())
