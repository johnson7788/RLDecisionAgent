#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:30
# @File  : train.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : PPT生成json大纲，然后按大纲数组的每个步骤进行搜索，即得到一个PPT的完整内容
"""
训练一个使用 LangGraph Tools 的 ReAct Agent（基于 ART 的 GRPO 强化训练）
- 工具：的网页搜索 + “提交最终答案”工具
- 训练：迭代采样 -> 组内相对打分(RULER, 可选) -> model.train()

依赖：
  uv pip install -U "openpipe-art[backend,langgraph]>=0.4.9" langchain-core pydantic tenacity litellm
  pip install wandb
"""

import os
import uuid
import time
import asyncio
from statistics import mean
from textwrap import dedent
from typing import List, Optional

import dotenv
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

# ---------------- wandb: 导入与初始化参数 ----------------
import wandb

dotenv.load_dotenv()

# ---------- 配置 ----------
# 任选一个可训练且支持 tools 的基础模型（Qwen2.5 系列在文档中常被用作示例）
NAME = os.getenv("ART_NAME", "web-search")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "web-search-agent-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"

# RULER 评估模型（可选；需相应 API Key）
RULER_MODEL = os.getenv("RULER_MODEL", "openai/o4-mini")

# wandb 相关配置（可选）
WANDB_PROJECT = os.getenv("WANDB_PROJECT", PROJECT_NAME)
WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # 组织名称，可为空
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}")

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

# ---------- Agent 协作所需的数据结构 ----------
class PlanItem(BaseModel):
    id: int = Field(..., description="从1开始的顺序编号")
    title: str = Field(..., description="该页的标题，<=12个词")
    queries: List[str] = Field(..., description="用于该页检索的1-3个具体搜索query")

class Slide(BaseModel):
    title: str
    bullets: List[str] = Field(..., description="3-5条，15-30词/条，含关键事实、日期或数据")
    source_urls: List[str] = Field(..., description="2-4个支持该页要点的URL")

class PPTAnswer(FinalAnswer):
    """向后兼容：保留 answer/source_urls，同时新增 slides"""
    slides: List[Slide] = Field(default_factory=list)


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
        base_url="http://127.0.0.1:6688",
        messages=messages,
        response_format=CorrectnessJudgeResponse,
    )
    return CorrectnessJudgeResponse.model_validate_json(
        resp.choices[0].message.content or "{}"
    )

# ---------- rollout：双 Agent（Planner + Researcher） ----------
async def rollout(model: art.Model, web_search_scenario: WebSearchScenario) -> ProjectTrajectory:
    scenario = web_search_scenario.scenario
    MAX_TURNS = 5

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": scenario.id, "step": web_search_scenario.step},
    )

    # ===== Agent 1：Planner（无工具，仅产出严格 JSON 大纲）=====
    planner_system = dedent("""
    You are a planning agent for creating a presentation outline.
    Given a presentation title, output a STRICT JSON array. Each item must be:
      { "id": <int starting from 1>, "title": "<<=12 words>", "queries": ["<search query 1>", "..."] }
    - Provide 4-6 items covering the topic from overview to specifics (consider adding a Conclusion).
    - Return ONLY raw JSON. No markdown, no prose.
    """)

    planner_model = init_chat_model(MODEL_NAME, temperature=0.2)

    # 产出大纲 JSON
    plan_msg = await planner_model.ainvoke(
        [SystemMessage(content=planner_system),
         HumanMessage(content=f"Presentation title: {scenario.question}")]
    )
    plan_text = (getattr(plan_msg, "content", None) or "").strip()

    # 容错解析（截取首尾方括号）
    import json, re
    def _parse_plan(txt: str) -> List[PlanItem]:
        try:
            data = json.loads(txt)
        except Exception:
            m = re.search(r"\[.*\]", txt, flags=re.S)
            data = json.loads(m.group(0)) if m else []
        # 兜底：若解析失败或为空，则给一个最小计划
        if not isinstance(data, list) or not data:
            data = [
                {"id": 1, "title": scenario.question, "queries": [scenario.question]},
                {"id": 2, "title": "Key Facts", "queries": [f"{scenario.question} key facts"]},
                {"id": 3, "title": "Conclusion", "queries": [f"{scenario.question} summary"]},
            ]
        # 规范化
        items = []
        for i, it in enumerate(data, start=1):
            items.append(PlanItem(
                id=int(it.get("id", i)),
                title=str(it.get("title", f"Slide {i}")),
                queries=[str(q) for q in (it.get("queries") or [])][:3] or [scenario.question]
            ))
        return items

    plan_items = _parse_plan(plan_text)

    # ===== Agent 2：Researcher（ReAct + 工具；逐项写页）=====
    slides: List[Slide] = []

    @tool
    async def web_search_tool(query: str) -> List[dict]:
        """Search the web by a query and return a list of results."""
        results = await search_web(query)
        return [r.model_dump() for r in results]

    @tool
    def return_slide_tool(title: str, bullets: List[str], source_urls: List[str]) -> dict:
        """Return one slide's content."""
        nonlocal slides
        slide = Slide(title=title, bullets=bullets, source_urls=source_urls)
        slides.append(slide)
        return slide.model_dump()

    @tool
    def return_final_answer_tool(answer: str, source_urls: List[str]) -> dict:
        """Return final answer with source URLs."""
        # 这里先占位，最终在函数末尾真正赋值；保持工具签名与原版兼容
        return {"answer": answer, "source_urls": source_urls}

    researcher_tools = [web_search_tool, return_slide_tool, return_final_answer_tool]
    researcher_system = dedent("""
    You are the Research & Write agent. For EACH slide:
    - Use web_search_tool with the given queries to gather facts.
    - Produce 3-5 concise, factual bullets (15-30 words each). Include specific data/dates.
    - Collect 2-4 reliable source URLs that support the bullets.
    - When done for THIS slide ONLY, call return_slide_tool(title, bullets, source_urls).
    Rules:
    - Prefer authoritative sources; avoid duplicates.
    - Do NOT call return_final_answer_tool here.
    """)

    researcher_model = init_chat_model(MODEL_NAME, temperature=0.7)
    researcher_agent = create_react_agent(researcher_model, researcher_tools)

    # 逐项执行：每个 PlanItem 作为独立的“写页”任务
    for item in plan_items:
        user_payload = dedent(f"""
        Slide #{item.id}: {item.title}
        Primary queries (use as you see fit):
        {chr(10).join('- ' + q for q in item.queries)}
        """)
        await researcher_agent.ainvoke(
            {"messages": [SystemMessage(content=researcher_system),
                          HumanMessage(content=user_payload)]},
            config={"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 6},
        )

    # ===== Finalizer：基于已生成的 slides，给出简短结论（用于训练打分的一句话答案）=====
    finalizer_model = init_chat_model(MODEL_NAME, temperature=0.0)
    slides_text = "\n\n".join(
        f"### {s.title}\n" + "\n".join(f"- {b}" for b in s.bullets) for s in slides
    )
    finalizer_sys = "You summarize the slides into ONE short, direct answer to the original question."
    final_msg = await finalizer_model.ainvoke(
        [SystemMessage(content=finalizer_sys),
         HumanMessage(content=f"Question: {scenario.question}\n\nSlides:\n{slides_text}\n\nAnswer in one sentence:")]
    )
    short_answer = (getattr(final_msg, "content", None) or "").strip()

    # 汇总所有来源（去重）
    all_srcs = []
    for s in slides:
        for u in s.source_urls:
            if u not in all_srcs:
                all_srcs.append(u)

    # 写回 Trajectory
    final_ans = PPTAnswer(answer=short_answer, source_urls=all_srcs, slides=slides)
    traj.final_answer = final_ans

    # 可选：简单正确性评估（保持你原逻辑）
    try:
        judge = await judge_correctness(scenario, final_ans.answer)
        traj.metrics["correct"] = float(judge.accept)
    except Exception:
        pass

    return traj

# ---------------- wandb: 封装一次性日志函数 ----------------
def _log_batch_to_wandb(*, batch, finished_groups, use_ruler: bool):
    """
    根据 gather_trajectory_groups 的结果整理并写入 wandb。
    兼容不同 art 版本中 TrajectoryGroup 的访问方式。
    """
    # 尝试尽可能地展开所有轨迹
    trajectories = []
    for g in finished_groups:
        # 优先使用属性
        if hasattr(g, "trajectories"):
            trajectories.extend(getattr(g, "trajectories"))
        else:
            # 某些实现中 group 可能是可迭代的
            try:
                trajectories.extend(list(g))
            except Exception:
                pass

    num_traj = len(trajectories)
    num_with_final = sum(1 for t in trajectories if getattr(t, "final_answer", None))
    correct_vals = []
    for t in trajectories:
        m = getattr(t, "metrics", None)
        if isinstance(m, dict) and "correct" in m:
            try:
                correct_vals.append(float(m["correct"]))
            except Exception:
                pass

    correct_rate = mean(correct_vals) if correct_vals else 0.0
    coverage = (num_with_final / num_traj) if num_traj else 0.0

    # 生成一个表用于检查 rollouts 的样例
    try:
        table = wandb.Table(columns=["scenario_id", "question", "ref_answer", "final_answer", "sources"])
        for t in trajectories[:50]:  # 避免过大
            meta = getattr(t, "metadata", {}) or {}
            s_id = meta.get("scenario_id", "")
            # 我们把 question/ref 从 messages 或 meta 中尽量拿到；若拿不到就跳过
            q = ""
            ref = ""
            try:
                # 在本脚本中，来源在 batch.items 中；构个索引
                for s in batch.items:
                    if s.id == s_id:
                        q, ref = s.question, s.answer
                        break
            except Exception:
                pass
            fa = getattr(t, "final_answer", None)
            ans = getattr(fa, "answer", "") if fa else ""
            srcs = ", ".join(getattr(fa, "source_urls", []) if fa else [])
            table.add_data(s_id, q, ref, ans, srcs)
    except Exception:
        table = None

    log_dict = {
        "train/step": batch.step,
        "train/epoch": batch.epoch,
        "ruler/enabled": int(bool(use_ruler)),
        "data/num_trajectories": num_traj,
        "data/final_answer_coverage": coverage,
        "eval/simple_correct_rate": correct_rate,
    }
    if table is not None:
        log_dict["samples/rollouts"] = table

    # 注意：设置 step 以便曲线按 step 展示
    wandb.log(log_dict, step=batch.step)


# ---------- 训练主程序 ----------
async def main():
    # ---------------- wandb: 初始化 ----------------
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY if WANDB_ENTITY else None,
        name=WANDB_RUN_NAME,
        config={
            "art_project": PROJECT_NAME,
            "art_name": NAME,
            "base_model": MODEL_NAME,
            "backend": "local" if USE_LOCAL_BACKEND else "skypilot",
            "ruler_model": RULER_MODEL,
        },
        settings=wandb.Settings(start_method="thread"),  # 与 asyncio 更兼容
    )
    # 让所有指标使用统一的 step
    wandb.define_metric("*", step_metric="train/step")

    # 选择后端
    if USE_LOCAL_BACKEND:
        from art.local.backend import LocalBackend
        backend = LocalBackend()
    else:
        from art.skypilot.backend import SkyPilotBackend
        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=os.getenv("ART_SKYPILOT_CLUSTER", "art-cluster"),
            gpu=os.getenv("ART_GPU", "A100"),
        )

    # 声明/注册模型
    model = art.TrainableModel(name=NAME, project=PROJECT_NAME, base_model=MODEL_NAME)
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

    # ---------------- wandb: 记录超参与数据概览 ----------------
    wandb.config.update(training_config)
    try:
        scen_table = wandb.Table(columns=["id", "question", "ref_answer"])
        for s in training_scenarios:
            scen_table.add_data(s.id, s.question, s.answer)
        wandb.log({"data/training_scenarios": scen_table}, step=0)
    except Exception:
        pass

    # 训练数据迭代器
    it = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    # 是否使用 RULER（无 judge key 时自动回退）
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
                    wrap_rollout(model, rollout)(model, WebSearchScenario(step=batch.step, scenario=s))
                    for _ in range(training_config["rollouts_per_group"])
                )
            )

        # 收集轨迹
        finished = await art.gather_trajectory_groups(
            groups, pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        # ---------------- wandb: 收集到的 rollouts 即刻记录 ----------------
        _log_batch_to_wandb(batch=batch, finished_groups=finished, use_ruler=use_ruler)

        # 打分 & 训练
        if use_ruler:
            extra_litellm_params = {"api_base": "http://localhost:6688", "api_key": os.environ["OPENAI_API_KEY"]}
            judged = []
            for g in finished:
                jg = await ruler_score_group(g, RULER_MODEL, extra_litellm_params=extra_litellm_params, debug=True)
                judged.append(jg)

            # 注意：art 的 train 返回值在不同版本可能不同，这里不强依赖返回值，只在 wandb 中标注一次
            await model.train(
                judged,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
                _config={"logprob_calculation_chunk_size": 8},
            )

            wandb.log({"train/used_judged_groups": 1}, step=batch.step)
        else:
            await model.train(
                finished,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            )
            wandb.log({"train/used_judged_groups": 0}, step=batch.step)

        if batch.step >= training_config["max_steps"]:
            break

    # ---------------- wandb: 结束 ----------------
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
