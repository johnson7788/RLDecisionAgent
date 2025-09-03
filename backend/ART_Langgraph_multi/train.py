#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:30
# @File  : train.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : PPT生成json大纲，然后按大纲数组的每个步骤进行搜索，即得到一个PPT的完整内容
"""
训练一个使用 LangGraph Tools 的 ReAct Agent（基于 ART 的 GRPO 强化训练）
- 工具：网页搜索 + “提交最终答案”工具
- 训练：迭代采样 -> 结构型奖励 -> (可选) RULER 打分 -> 加权融合 -> model.train()

依赖：
  uv pip install -U "openpipe-art[backend,langgraph]>=0.4.9" langchain-core pydantic tenacity litellm
  pip install wandb
"""
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import uuid
import time
import asyncio
import json
import re
from statistics import mean
from textwrap import dedent
from typing import List, Optional, Tuple, Dict

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
from urllib.parse import urlparse
from zai import ZhipuAiClient

# ---------------- wandb: 导入与初始化参数 ----------------
import wandb

dotenv.load_dotenv()

# ---------- 配置 ----------
NAME = os.getenv("ART_NAME", "web-search")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "web-search-agent-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"

# RULER 评估模型（可选；需相应 API Key）
RULER_MODEL = os.getenv("RULER_MODEL", "openai/o4-mini")
ALPHA_RULER = float(os.getenv("ALPHA_RULER", "0.6"))  # 融合权重：最终reward = alpha*RULER + (1-alpha)*Structure

print(f"{NAME} - {MODEL_NAME} - {PROJECT_NAME} - {os.environ['WANDB_BASE_URL']}")


# 结构型奖励权重（总和建议=1）
STRUCT_WEIGHTS = {
    "slides_count": 0.20,
    "bullets_per_slide": 0.20,
    "bullet_length": 0.20,
    "numbers_dates": 0.15,
    "sources_count": 0.15,
    "sources_unique": 0.05,
    "source_quality": 0.05,
}
WANDB_PROJECT = os.getenv("WANDB_PROJECT", PROJECT_NAME)
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
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
    question: str  # 用作“演示标题”
    answer: str    # 一句话论点（Finalizer 风格对齐）


class WebSearchScenario(BaseModel):
    step: int
    scenario: Scenario


class ProjectTrajectory(art.Trajectory):
    final_answer: Optional["PPTAnswer"] = None


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


# ---------- 工具函数：检索 ----------
async def search_web(keyword: str) -> List[WebSearchResult]:
    response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=15,
        search_recency_filter="noLimit",
        content_size="high",
    )
    return [
        WebSearchResult(
            url=item["url"],
            title=item["title"],
            snippet=item["content"],
        )
        for item in response["search_result"]
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


# ---------- 结构型奖励计算 ----------
GOOD_DOMAINS = [
    "who.int", "un.org", "oecd.org", "nature.com", "science.org", "ft.com",
    "bbc.com", "nytimes.com", "reuters.com", "ec.europa.eu", "nasa.gov",
    "noaa.gov",
]


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return ""


def _score_range_closed(minv: int, maxv: int, val: int, soft_center: Optional[int] = None) -> float:
    """在[minv,maxv]给满分1；否则线性衰减，软中心可增强中心附近权重"""
    if val is None:
        return 0.0
    if minv <= val <= maxv:
        return 1.0
    # 距离越远分越低；以区间宽度为尺度
    width = max(1, maxv - minv)
    if val < minv:
        dist = minv - val
    else:
        dist = val - maxv
    base = max(0.0, 1.0 - dist / (width * 2))
    if soft_center is not None and minv <= soft_center <= maxv:
        # 在中心附近略微抬升（可选）
        center_boost = max(0.0, 1.0 - abs(val - soft_center) / width)
        base = max(base, 0.5 * base + 0.5 * center_boost)
    return float(base)


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DIGIT_RE = re.compile(r"\d")


def _bullet_len_score(text: str) -> float:
    """优先按词数(15-30)；若词数<2（如纯中文），则按字符长度(20-80)"""
    words = _WORD_RE.findall(text)
    if len(words) >= 2:
        return _score_range_closed(15, 30, len(words))
    # fallback: char length
    chars = len(text.strip())
    return _score_range_closed(20, 80, chars)


def _source_quality_score(urls: List[str]) -> float:
    if not urls:
        return 0.0
    good = 0
    for u in urls:
        h = _host(u)
        if h.endswith(".gov") or h.endswith(".edu"):
            good += 1
        elif any(k in h for k in GOOD_DOMAINS):
            good += 1
    return min(1.0, good / max(1, len(urls)))  # 比例


def compute_structure_reward(slides: List[Slide]) -> Tuple[float, Dict[str, float]]:
    """返回 (总体结构分, 各子项分)；范围[0,1]"""
    if not slides:
        return 0.0, {k: 0.0 for k in STRUCT_WEIGHTS}

    # 1) 页数
    n_slides = len(slides)
    s_slides = _score_range_closed(4, 6, n_slides, soft_center=5)

    # 2) 每页要点数（3-5）
    bullet_counts = [len(s.bullets or []) for s in slides]
    per_ok = [1.0 if 3 <= c <= 5 else _score_range_closed(3, 5, c) for c in bullet_counts]
    s_bullets_per_slide = sum(per_ok) / n_slides

    # 3) 要点长度（15-30词 或 20-80字）
    all_bullets = [b for s in slides for b in (s.bullets or [])]
    if all_bullets:
        s_bullet_len = sum(_bullet_len_score(b) for b in all_bullets) / len(all_bullets)
    else:
        s_bullet_len = 0.0

    # 4) 含数字/日期比例
    if all_bullets:
        with_num = 0
        for b in all_bullets:
            if _DIGIT_RE.search(b) or _YEAR_RE.search(b):
                with_num += 1
        ratio = with_num / len(all_bullets)
        s_numbers_dates = min(1.0, ratio / 0.5)  # ≥50%给满分
    else:
        s_numbers_dates = 0.0

    # 5) 来源数量（每页 2-4）
    src_counts = [len(s.source_urls or []) for s in slides]
    src_ok = [1.0 if 2 <= c <= 4 else _score_range_closed(2, 4, c) for c in src_counts]
    s_sources_count = sum(src_ok) / n_slides

    # 6) 来源唯一性（去重率）
    all_src = [u for s in slides for u in (s.source_urls or [])]
    if all_src:
        unique_src = list(dict.fromkeys(all_src))
        s_sources_unique = len(unique_src) / len(all_src)
    else:
        s_sources_unique = 0.0

    # 7) 来源权威度（按每页均值）
    if slides:
        s_source_quality = sum(_source_quality_score(s.source_urls or []) for s in slides) / n_slides
    else:
        s_source_quality = 0.0

    subs = {
        "slides_count": s_slides,
        "bullets_per_slide": s_bullets_per_slide,
        "bullet_length": s_bullet_len,
        "numbers_dates": s_numbers_dates,
        "sources_count": s_sources_count,
        "sources_unique": s_sources_unique,
        "source_quality": s_source_quality,
    }

    # 归一化权重
    wsum = sum(max(0.0, float(w)) for w in STRUCT_WEIGHTS.values()) or 1.0
    weights = {k: max(0.0, float(STRUCT_WEIGHTS.get(k, 0.0))) / wsum for k in subs.keys()}

    total = sum(subs[k] * weights[k] for k in subs.keys())
    return float(total), {**subs, **{f"w_{k}": weights[k] for k in subs.keys()}}


# ---------- rollout：双 Agent（Planner + Researcher） ----------
async def rollout(model: art.Model, web_search_scenario: WebSearchScenario) -> ProjectTrajectory:
    scenario = web_search_scenario.scenario

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

    plan_msg = await planner_model.ainvoke(
        [SystemMessage(content=planner_system),
         HumanMessage(content=f"Presentation title: {scenario.question}")]
    )
    plan_text = (getattr(plan_msg, "content", None) or "").strip()

    # 容错解析（截取首尾方括号）
    def _parse_plan(txt: str) -> List[PlanItem]:
        try:
            data = json.loads(txt)
        except Exception:
            m = re.search(r"\[.*\]", txt, flags=re.S)
            data = json.loads(m.group(0)) if m else []
        if not isinstance(data, list) or not data:
            data = [
                {"id": 1, "title": scenario.question, "queries": [scenario.question]},
                {"id": 2, "title": "Key Facts", "queries": [f"{scenario.question} key facts"]},
                {"id": 3, "title": "Conclusion", "queries": [f"{scenario.question} summary"]},
            ]
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

    # 逐页执行
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

    # ===== Finalizer：从 slides 生成一句话论点（与标注风格对齐）=====
    finalizer_model = init_chat_model(MODEL_NAME, temperature=0.0)
    slides_text = "\n\n".join(
        f"### {s.title}\n" + "\n".join(f"- {b}" for b in s.bullets) for s in slides
    )
    finalizer_sys = (
        "You write ONE crisp thesis sentence that directly states the key takeaway "
        "of the slides, without prefixes like 'In summary' or hedging language."
    )
    final_msg = await finalizer_model.ainvoke(
        [SystemMessage(content=finalizer_sys),
         HumanMessage(content=f"Presentation title: {scenario.question}\n\nSlides:\n{slides_text}\n\nThesis (one sentence):")]
    )
    short_answer = (getattr(final_msg, "content", None) or "").strip()

    # 汇总所有来源（去重）
    all_srcs = []
    for s in slides:
        for u in s.source_urls:
            if u not in all_srcs:
                all_srcs.append(u)

    final_ans = PPTAnswer(answer=short_answer, source_urls=all_srcs, slides=slides)
    traj.final_answer = final_ans

    # ===== 结构型奖励计算 =====
    struct_score, subs = compute_structure_reward(slides)
    traj.reward = float(struct_score)
    # 写入 metrics（便于 wandb 分项观测）
    traj.metrics = getattr(traj, "metrics", {}) or {}
    traj.metrics["structure_score"] = struct_score
    for k, v in subs.items():
        traj.metrics[f"structure/{k}"] = float(v)

    # 可选：简单正确性评估（与数据集中一句话论点对齐）
    try:
        judge = await judge_correctness(scenario, final_ans.answer)
        traj.metrics["correct"] = float(judge.accept)
    except Exception:
        pass

    return traj


# ---------------- wandb: 日志封装 ----------------
def _log_batch_to_wandb(*, batch, finished_groups, use_ruler: bool):
    trajectories = []
    for g in finished_groups:
        if hasattr(g, "trajectories"):
            trajectories.extend(getattr(g, "trajectories"))
        else:
            try:
                trajectories.extend(list(g))
            except Exception:
                pass

    num_traj = len(trajectories)
    num_with_final = sum(1 for t in trajectories if getattr(t, "final_answer", None))

    correct_vals, struct_vals = [], []
    subs_avg: Dict[str, List[float]] = {}
    for t in trajectories:
        m = getattr(t, "metrics", None)
        if isinstance(m, dict):
            if "correct" in m:
                try: correct_vals.append(float(m["correct"]))
                except Exception: pass
            if "structure_score" in m:
                try: struct_vals.append(float(m["structure_score"]))
                except Exception: pass
            for k, v in m.items():
                if k.startswith("structure/") and not k.startswith("structure/w_"):
                    subs_avg.setdefault(k, []).append(float(v))

    correct_rate = mean(correct_vals) if correct_vals else 0.0
    struct_mean = mean(struct_vals) if struct_vals else 0.0
    coverage = (num_with_final / num_traj) if num_traj else 0.0

    # rollouts 样例表
    try:
        table = wandb.Table(columns=["scenario_id", "question", "ref_answer", "final_answer", "sources"])
        for t in trajectories[:50]:
            meta = getattr(t, "metadata", {}) or {}
            s_id = meta.get("scenario_id", "")
            q = ""
            ref = ""
            try:
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
        "reward/structure_mean": struct_mean,
    }
    if table is not None:
        log_dict["samples/rollouts"] = table
    # 子项平均
    for k, arr in subs_avg.items():
        log_dict[f"reward/{k}_mean"] = mean(arr) if arr else 0.0

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
            "alpha_ruler": ALPHA_RULER,
            "struct_weights": STRUCT_WEIGHTS,
        },
        settings=wandb.Settings(start_method="thread"),
    )
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

    # ========== 训练集 ==========
    training_scenarios = [
        Scenario(id="p1",
            question="PPT 生成 Agent 的双角色协作范式（Planner·Researcher）",
            answer="将规划与检索写作解耦，能稳定提升结构完整性与可验证性，比单体大模型更可靠。"),
        Scenario(id="p2",
            question="从大纲到成稿：检索驱动的演示文稿工作流",
            answer="标准化大纲、可追溯引用与逐页写作循环是生成高质量演示文稿的核心。"),
        Scenario(id="p3",
            question="RAG 与传统网页检索在生成演示中的分工",
            answer="网页检索覆盖广度与时效，RAG提供域内深度与一致性，二者互补最优。"),
        Scenario(id="p4",
            question="信息源可信度与引用规范在演示中的作用",
            answer="来源权威性与可复核性直接决定演示的说服力与可传播性。"),
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

    # ------ 融合函数：把 RULER reward 与结构分融合 ------
    def blend_rewards_inplace(orig_groups, judged_groups, alpha: float):
        for og, jg in zip(orig_groups, judged_groups):
            try:
                olist = getattr(og, "trajectories") if hasattr(og, "trajectories") else list(og)
                jlist = getattr(jg, "trajectories") if hasattr(jg, "trajectories") else list(jg)
                for ot, jt in zip(olist, jlist):
                    struct_r = float(getattr(ot, "reward", 0.0) or 0.0)  # rollout 中的结构分
                    ruler_r = float(getattr(jt, "reward", 0.0) or 0.0)   # RULER 打分
                    jt.reward = alpha * ruler_r + (1.0 - alpha) * struct_r
            except Exception:
                continue

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

        # 收集轨迹（此时 reward=结构分）
        finished = await art.gather_trajectory_groups(
            groups, pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        # 记录 rollouts 与结构分
        _log_batch_to_wandb(batch=batch, finished_groups=finished, use_ruler=use_ruler)

        # 打分 & 训练
        if use_ruler:
            extra_litellm_params = {"api_base": "http://localhost:6688", "api_key": os.environ.get("OPENAI_API_KEY", "")}
            judged = []
            for g in finished:
                jg = await ruler_score_group(g, RULER_MODEL, extra_litellm_params=extra_litellm_params, debug=True)
                judged.append(jg)

            # 融合：alpha*RULER + (1-alpha)*STRUCTURE
            blend_rewards_inplace(finished, judged, ALPHA_RULER)

            await model.train(
                judged,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
                _config={"logprob_calculation_chunk_size": 8},
            )
            wandb.log({"train/used_judged_groups": 1, "reward/alpha_ruler": ALPHA_RULER}, step=batch.step)
        else:
            # 无 RULER：直接用结构分作为 reward
            await model.train(
                finished,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            )
            wandb.log({"train/used_judged_groups": 0}, step=batch.step)

        if batch.step >= training_config["max_steps"]:
            break

    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
