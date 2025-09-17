#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27
# @File  : train_query_agent.py
# @Author: johnson
# @Desc  : 训练一个“查询”ReAct Agent（ART + LangGraph，GRPO），从 questions.txt 读取问题
#          工具：检索/计算等来自 MCP 服务器；return_final_answer_tool 使用本地工具以便训练阶段读取 final_answer。

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
from langchain_core.tools import Tool, tool
from pydantic import BaseModel, Field, ValidationError

from reward import search_reward, format_reward

# ==== MCP ====
# pip install fastmcp
from fastmcp import Client as MCPClient
from mcpserver.mcp_client import tool_definition_to_dict  #自定义MCP工具

dotenv.load_dotenv()

# ---------------- 运行配置 ----------------
NAME = os.getenv("ART_NAME", "query-web")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "content-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"

WANDB_PROJECT = os.getenv("WANDB_PROJECT", PROJECT_NAME)
WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # 可空
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}")
MCP_CONFIG = os.getenv("MCP_CONFIG", "mcp_config.json")

print(f"{NAME} - {MODEL_NAME} - {PROJECT_NAME} - {os.getenv('WANDB_BASE_URL', '')}")
print(f"使用MCP的配置文件: {MCP_CONFIG}")

# ----------------- 数据结构 -----------------
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

# =========================
# MCP: 发现与调用
# =========================

class MCPRegistry:
    """保存已发现的 MCP 工具（JSON Schema）以及 名称→服务器URL 映射。"""
    def __init__(self):
        self.tools_schema: List[Dict[str, Any]] = []
        self.name_to_server: Dict[str, str] = {}

    def tools_schema_json(self) -> str:
        return json.dumps(self.tools_schema, ensure_ascii=False, indent=2)

    def find_server(self, tool_name: str) -> Optional[str]:
        return self.name_to_server.get(tool_name)

async def discover_mcp_tools(config_path: str) -> MCPRegistry:
    """从 JSON 配置读取 MCP 服务器，并汇总工具定义。"""
    reg = MCPRegistry()
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"MCP配置文件不存在: {config_path}")

    mcp_servers = (cfg or {}).get("mcpServers", {})
    if not mcp_servers:
        return reg

    for server_name, info in mcp_servers.items():
        if info.get("disabled"):
            continue
        url = info.get("url")
        if not url:
            continue
        try:
            client = MCPClient(url)
            async with client:
                tools = await client.list_tools()
        except Exception as e:
            print(f"⚠️  无法从 '{server_name}' ({url}) 获取工具: {e}")
            continue

        for t in tools:
            try:
                d = tool_definition_to_dict(t)  # {name, description, parameters}
            except Exception:
                d = {"name": getattr(t, "name", ""), "description": getattr(t, "description", None)}
            name = d.get("name")
            if not name:
                continue

            # 跳过 MCP 端的 return_final_answer_tool，确保使用本地版本
            if name == "return_final_answer_tool":
                print("ℹ️  跳过 MCP 端 return_final_answer_tool（使用本地实现用于训练）。")
                continue

            if name in reg.name_to_server:
                print(f"⚠️  工具名冲突: '{name}' 在多个服务器中发现，将使用先发现的服务器 {reg.name_to_server[name]}")
            else:
                reg.name_to_server[name] = url

            if "parameters" not in d or d["parameters"] is None:
                d["parameters"] = {"type": "object", "properties": {}, "required": []}
            reg.tools_schema.append(d)

    print(f"🔧 通过 MCP 发现 {len(reg.tools_schema)} 个工具（不含本地 return_final_answer_tool）。")
    return reg

async def mcp_call_tool(server_url: str, tool_name: str, arguments: Dict[str, Any] | None) -> Any:
    """在指定 MCP 服务器上调用工具，并返回 JSON 可序列化结果。"""
    arguments = arguments or {}
    client = MCPClient(server_url)
    try:
        async with client:
            try:
                result = await client.call_tool(tool_name, arguments)
            except AttributeError:
                result = await client.call(tool_name, arguments)
    except Exception as e:
        return {"error": f"调用工具失败: {e}"}

    # 归一化
    def _normalize(x: Any) -> Any:
        try:
            if hasattr(x, "model_dump"):
                x = x.model_dump()
        except Exception:
            pass
        try:
            json.dumps(x, ensure_ascii=False)
            return x
        except TypeError:
            return repr(x)

    return _normalize(result)

def _parse_json_obj_loose(s: Any) -> Optional[Dict[str, Any]]:
    """宽松解析：输入可以是 dict / JSON 字符串 / ```json fenced。"""
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    t = s.strip()
    if t.startswith("```") and t.endswith("```"):
        inner = t.strip("`")
        # 去掉可能的语言标签
        for tag in ("json", "jsonc", "javascript", "js", "txt"):
            if inner.lower().startswith(tag):
                inner = inner[len(tag):].strip()
                break
        t = inner
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        import re
        m = re.search(r"\{(?:.|\s)*\}", t)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None

def _build_tool_description(defn: Dict[str, Any]) -> str:
    desc = (defn.get("description") or "").strip()
    schema = defn.get("parameters") or {}
    return dedent(f"""
    [MCP] {desc}
    参数 JSON Schema（供参考）：
    {json.dumps(schema, ensure_ascii=False, indent=2)}
    """).strip()

def _make_langchain_tool_for_mcp(tool_name: str, server_url: str, defn: Dict[str, Any]) -> Tool:
    """把单个 MCP 工具包装成 LangChain Tool（仅做转发，不含业务逻辑）。"""
    class _Args(BaseModel):
        input: str = Field(description="参数对象的 JSON 字符串，例如 '{\"q\":\"北京天气\"}'")

    async def _acoroutine(input: str) -> str:
        args = _parse_json_obj_loose(input) or {}
        res = await mcp_call_tool(server_url, tool_name, args)
        try:
            return json.dumps(res, ensure_ascii=False)
        except Exception:
            return str(res)

    return Tool(
        name=tool_name,
        description=_build_tool_description(defn),
        args_schema=_Args,
        coroutine=_acoroutine,
    )

# ----------------- 从文件加载问题 -----------------
def _default_questions_path() -> str:
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
            if s.startswith("#"):
                continue
            questions.append(s)
    if not questions:
        raise ValueError("questions.txt 为空或没有有效问题行。")
    return questions

# ----------------- rollout（核心）：LangGraph + MCP 工具 + 本地 return_final_answer_tool -----------------
async def rollout(model: art.Model, scenario: QueryScenario) -> ProjectTrajectory:
    MAX_TURNS = 16
    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": scenario.id}
    )

    # 发现 MCP 工具并注册为 LangChain 工具（不包含 return_final_answer_tool）
    registry = await discover_mcp_tools(MCP_CONFIG)
    lc_tools: List[Tool] = []
    for defn in registry.tools_schema:
        name = defn.get("name")
        if not name:
            continue
        server_url = registry.find_server(name)
        if not server_url:
            continue
        lc_tools.append(_make_langchain_tool_for_mcp(name, server_url, defn))

    # 本地：return_final_answer_tool（用于训练阶段保存最终 JSON）
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

    lc_tools.append(return_final_answer_tool)

    if not lc_tools:
        raise RuntimeError("未从 MCP 发现任何可用工具，请先启动 MCP 服务器并检查配置。")

    # ====== 提示词：查询 Agent 要求 ======
    tools_json = registry.tools_schema_json() + "\n(另含本地：return_final_answer_tool(task, sources))"
    system_prompt = dedent(f"""
    你是一个数据查询与分析助手（Query Agent）。你的任务：
    1) 读取用户提供的单元素 JSON 任务数组（task），其中包含一个 {{type:"qa", data:{{question, text}}}}；
    2) 必要时调用**已发现的 MCP 工具**进行检索/计算/转换；这些工具在本训练中统一只有一个参数 input（字符串），你需要传入参数对象的 JSON 字符串；
    3) 输出 2~6 句，包含可核验事实（数字/日期/机构/地名等），并在句尾添加来源引用 [n]（n 从 1 开始，对应 sources 中 URL 的顺序）；
    4) 不得修改输入 JSON 的结构和字段，只能用最终答案覆盖 data.text；
    5) 完成后**必须调用本地的 `return_final_answer_tool(task, sources)`** 返回最终 JSON：
       - task：保持与输入一致，仅把 data.text 替换为你的答案（包含 [n] 引用）；
       - sources：去重后的 URL 列表，顺序与 [n] 对应；
    6) 不要在普通对话中粘贴 JSON，务必通过工具返回最终 JSON。

    下面是已发现的 MCP 工具（JSON Schema）：\n{tools_json}
    """)

    chat_model = init_chat_model(MODEL_NAME, temperature=0.8)
    agent = create_react_agent(chat_model, tools=lc_tools)

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
    # 收集工具阶段见过的 URL，用于一致性校验（对所有工具结果尝试抽取）
    tool_urls_seen: List[str] = []
    try:
        for m in traj.messages_and_choices:
            if m.get("role") != "tool":
                continue
            content = m.get("content")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    content = _parse_json_obj_loose(content)

            def _harvest(obj: Any):
                if isinstance(obj, list):
                    for it in obj:
                        if isinstance(it, dict) and isinstance(it.get("url"), str):
                            tool_urls_seen.append(it["url"])
                elif isinstance(obj, dict):
                    if isinstance(obj.get("url"), str):
                        tool_urls_seen.append(obj["url"])
                    for v in obj.values():
                        _harvest(v)
            _harvest(content)
    except Exception:
        pass

    if final:
        traj.final = final

        # 仍然沿用原奖励：格式 + 搜索一致性
        fr = 0.0
        try:
            fr = format_reward(scenario.input_task, final.task)
        except Exception:
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
    else:
        # 未返回最终 JSON，给最低奖励
        traj.reward = 0.0
        traj.metrics["format_reward"] = 0.0
        traj.metrics["search_reward"] = 0.0
        traj.metrics["sources_count"] = 0

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
