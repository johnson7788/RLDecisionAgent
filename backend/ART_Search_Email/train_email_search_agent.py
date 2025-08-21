#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/21 09:46
# @File  : train_email_search_agent.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

功能：
1) 构建 Enron 邮件 SQLite 数据库（含 FTS5 全文索引）；
2) 使用 ART 训练一个“邮件搜索 Agent”（Qwen2.5-7B-Instruct 基座）；
3) 采用 RULER 相对评分作为奖励，异步训练循环。

说明：
- 所有需要 await 的操作均封装成 async 函数，通过 asyncio.run(...) 调用。
- 可执行入口仅在 __main__，因此该文件也可被测试脚本安全 import 复用核心函数。
"""

import argparse
import asyncio
import json
import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from textwrap import dedent
from typing import List, Literal, Optional

from dotenv import load_dotenv
from datasets import Dataset, Features, Sequence, Value, load_dataset
from pydantic import BaseModel, Field
from tqdm import tqdm
# ========== ART / 训练（全部异步封装） ==========

import art
import weave
from art.local import LocalBackend
from art.utils import iterate_dataset
from art.rewards import ruler_score_group
from art.utils.litellm import convert_litellm_choice_to_openai
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm import acompletion
from tenacity import retry, stop_after_attempt

MAX_TURNS = 10  # 单回合最多交互轮数

# ========== 数据模型与数据库 ==========

class Email(BaseModel):
    """单封邮件结构（与 DB 字段对应）"""
    message_id: str
    date: str  # 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []
    body: Optional[str] = None
    file_name: Optional[str] = None


class Scenario(BaseModel):
    """训练/评测场景"""
    id: int
    question: str
    answer: str
    message_ids: List[str]
    how_realistic: float
    inbox_address: str
    query_date: str
    split: Literal["train", "test"]


@dataclass
class SearchResult:
    """FTS 搜索返回值"""
    message_id: str
    snippet: str


class FinalAnswer(BaseModel):
    """Agent 最终回答及引用来源"""
    answer: str
    source_ids: list[str]


DB_PATH = "./enron_emails.db"
EMAIL_DATASET_REPO_ID = "corbt/enron-emails"
SCENARIO_DATASET_REPO_ID = "corbt/enron_emails_sample_questions"

_db_conn = None  # 全局连接（简化示例）


def create_email_database() -> sqlite3.Connection:
    """从 HF 构建 SQLite/FTS 数据库（耗时较久）"""
    print("Creating email database from Hugging Face dataset...")

    SQL_CREATE_TABLES = """
    DROP TABLE IF EXISTS recipients;
    DROP TABLE IF EXISTS emails_fts;
    DROP TABLE IF EXISTS emails;

    CREATE TABLE emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT UNIQUE,
        subject TEXT,
        from_address TEXT,
        date TEXT,
        body TEXT,
        file_name TEXT
    );

    CREATE TABLE recipients (
        email_id TEXT,
        recipient_address TEXT,
        recipient_type TEXT
    );
    """

    SQL_CREATE_INDEXES_TRIGGERS = """
    CREATE INDEX idx_emails_from ON emails(from_address);
    CREATE INDEX idx_emails_date ON emails(date);
    CREATE INDEX idx_emails_message_id ON emails(message_id);
    CREATE INDEX idx_recipients_address ON recipients(recipient_address);
    CREATE INDEX idx_recipients_type ON recipients(recipient_type);
    CREATE INDEX idx_recipients_email_id ON recipients(email_id);
    CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

    CREATE VIRTUAL TABLE emails_fts USING fts5(
        subject,
        body,
        content='emails',
        content_rowid='id'
    );

    CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
        INSERT INTO emails_fts (rowid, subject, body)
        VALUES (new.id, new.subject, new.body);
    END;
    CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
        DELETE FROM emails_fts WHERE rowid=old.id;
    END;
    CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
        UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
    END;
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()

    print("Loading Enron dataset from HF...")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )
    dataset = load_dataset(EMAIL_DATASET_REPO_ID, features=expected_features, split="train")
    print(f"Dataset contains {len(dataset)} total emails")

    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    record_count = skipped_count = duplicate_count = 0
    dedup = set()

    for email in tqdm(dataset, desc="Inserting emails"):
        message_id = email["message_id"]
        subject = email["subject"]
        from_address = email["from"]
        date_obj: datetime = email["date"]
        body = email["body"] or ""
        file_name = email["file_name"]
        to_list = [str(a) for a in email["to"] if a]
        cc_list = [str(a) for a in email["cc"] if a]
        bcc_list = [str(a) for a in email["bcc"] if a]

        if len(body) > 5000:
            skipped_count += 1
            continue
        if (len(to_list) + len(cc_list) + len(bcc_list)) > 30:
            skipped_count += 1
            continue

        key = (subject, body, from_address)
        if key in dedup:
            duplicate_count += 1
            continue
        dedup.add(key)

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO emails (message_id, subject, from_address, date, body, file_name) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, subject, from_address, date_str, body, file_name),
        )

        # Insert recipients
        recipient_data = []
        for addr in to_list:
            recipient_data.append((message_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((message_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((message_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
            """,
                recipient_data,
            )

        record_count += 1

    conn.commit()

    print("Creating indexes & FTS...")
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    cursor.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
    conn.commit()

    print(f"✅ DB ready: {record_count} emails | skipped {skipped_count} | dup {duplicate_count}")
    return conn


def get_db_connection() -> sqlite3.Connection:
    """单例连接"""
    global _db_conn
    if _db_conn is None:
        if os.path.exists(DB_PATH):
            print(f"Loading existing DB: {DB_PATH}")
            _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        else:
            _db_conn = create_email_database()
    return _db_conn


def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """FTS5 搜索主题/正文，并限定属于某 inbox（发件或收件包含该地址）"""
    if not keywords:
        raise ValueError("No keywords provided for search.")
    if max_results > 10:
        raise ValueError("max_results must be <= 10.")

    conn = get_db_connection()
    cursor = conn.cursor()

    where, params = [], []

    fts_query = " ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    where.append("""
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r WHERE r.recipient_address = ? AND r.email_id = e.message_id
        ))
    """)
    params.extend([inbox, inbox])

    if from_addr:
        where.append("e.from_address = ?")
        params.append(from_addr)
    if to_addr:
        where.append("""
            EXISTS (SELECT 1 FROM recipients r_to
                    WHERE r_to.recipient_address = ? AND r_to.email_id = e.message_id)
        """)
        params.append(to_addr)
    if sent_after:
        where.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")
    if sent_before:
        where.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
        SELECT e.message_id,
               snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) AS snippet
        FROM emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE {" AND ".join(where)}
        ORDER BY e.date DESC
        LIMIT ?;
    """
    params.append(max_results)

    cursor.execute(sql, params)
    results = cursor.fetchall()

    return [SearchResult(message_id=row[0], snippet=row[1]) for row in results]


def read_email(message_id: str) -> Optional[Email]:
    """按 message_id 读取完整邮件与收/抄/密送"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT message_id, date, subject, from_address, body, file_name FROM emails WHERE message_id = ?",
              (message_id,))
    row = c.fetchone()
    if not row:
        return None
    msg_id, date, subject, from_addr, body, file_name = row

    c.execute("SELECT recipient_address, recipient_type FROM recipients WHERE email_id = ?", (message_id,))
    to_addrs, cc_addrs, bcc_addrs = [], [], []
    for addr, typ in c.fetchall():
        t = (typ or "").lower()
        if t == "to": to_addrs.append(addr)
        elif t == "cc": cc_addrs.append(addr)
        elif t == "bcc": bcc_addrs.append(addr)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addrs,
        cc_addresses=cc_addrs,
        bcc_addresses=bcc_addrs,
        body=body,
        file_name=file_name,
    )


def load_training_scenarios(
    split: Literal["train", "test"] = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Scenario]:
    """从 HF 加载场景（可限制每场景引用的邮件数量）"""
    print(f"Loading {split} scenarios from HF...")
    dataset: Dataset = load_dataset(SCENARIO_DATASET_REPO_ID, split=split)

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if shuffle or (seed is not None):
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()

    # Convert each row to a Scenario object
    scenarios = [Scenario(**row, split=split) for row in dataset]

    if max_messages is not None:
        scenarios = [s for s in scenarios if len(s.message_ids) <= max_messages]

    if shuffle:
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(scenarios)
        else:
            random.shuffle(scenarios)

    if limit is not None:
        scenarios = scenarios[:limit]

    print(f"Loaded {len(scenarios)} scenarios.")
    return scenarios



class CorrectnessJudgeResponse(BaseModel):
    """裁判模型结构化返回"""
    reasoning: str = Field(description="Why accepted or rejected")
    accept: bool = Field(description="Accept this AI answer?")


@retry(stop=stop_after_attempt(3))
async def judge_correctness(scenario: Scenario, answer: str) -> CorrectnessJudgeResponse:
    """调用第三方裁判模型（通过 LiteLLM 接口）"""
    system_prompt = dedent("""
        You are given a question, a reference answer, and an AI answer.
        Accept the AI answer only if it contains the relevant information from the reference answer; otherwise reject.
    """)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {scenario.question}\nReference answer: {scenario.answer}\nAI answer: {answer}"},
    ]
    resp = await acompletion(
        model="openai/o4-mini",   # 可改为 openai/gpt-4.1 等
        base_url="http://127.0.0.1:6688",
        messages=messages,
        response_format=CorrectnessJudgeResponse,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return CorrectnessJudgeResponse.model_validate_json(raw)
    except Exception as e:
        return CorrectnessJudgeResponse(reasoning=f"Parse error: {e}\nRaw: {raw}", accept=False)


class ProjectTrajectory(art.Trajectory):
    """扩展 Trajectory 保存最终回答"""
    final_answer: FinalAnswer | None = None


class EmailScenario(BaseModel):
    """rollout 输入类型"""
    step: int
    scenario: Scenario


@weave.op
async def rollout(model: art.Model, email_scenario: EmailScenario) -> ProjectTrajectory:
    """
    单回合执行：让 Agent 使用工具搜索与阅读邮件，直到产出最终回答或轮次耗尽。
    （异步函数，供训练与测试复用）
    """
    scenario = email_scenario.scenario
    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": scenario.id, "step": email_scenario.step},
    )

    system_prompt = dedent(f"""
        You are an email search agent. Use the provided tools to search and read emails.
        You may take up to {MAX_TURNS} turns.
        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}
    """)

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]

    # ---- 工具定义（提供给 LLM 可调用） ----
    def search_inbox(keywords: list[str]) -> list[dict]:
        results = search_emails(
            inbox=scenario.inbox_address,
            keywords=keywords,
            sent_before=scenario.query_date,
        )
        return [asdict(r) for r in results]

    def return_final_answer(answer: str, reference_message_ids: list[str]) -> FinalAnswer:
        return FinalAnswer(answer=answer, source_ids=reference_message_ids)

    tools = [search_inbox, read_email, return_final_answer]
    tools_by_name = {t.__name__: t for t in tools}
    traj.tools = [convert_to_openai_tool(t) for t in tools]

    if model.trainable:
        litellm_model_name = f"hosted_vllm/{model.name}"
    else:
        litellm_model_name = model.name

    for _ in range(MAX_TURNS):
        resp = await acompletion(
            model=litellm_model_name,
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
            temperature=1,
            messages=traj.messages(),
            caching=False,
            tools=traj.tools,
        )
        choice = resp.choices[0]
        msg = choice.message
        traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))

        if not msg.tool_calls:
            return traj

        try:
            for tc in msg.tool_calls:
                name = tc.function.name
                if name in tools_by_name:
                    args = json.loads(tc.function.arguments)
                    result = tools_by_name[name](**args)
                    traj.messages_and_choices.append(
                        {"role": "tool", "tool_call_id": tc.id, "name": name, "content": str(result)}
                    )
                    if name == "return_final_answer":
                        traj.final_answer = result
                        judge = await judge_correctness(scenario, traj.final_answer.answer)
                        traj.metrics["correct"] = float(judge.accept)
                        return traj
        except Exception as e:
            print(f"Error parsing tool calls: {e}")
            return traj

    return traj


async def setup_model_and_backend(seed: int = 42) -> art.TrainableModel:
    """
    初始化可训练模型与本地 Backend（异步）。
    说明：把 await 相关的初始化单独封装，便于训练/测试脚本复用。
    """
    random.seed(seed)
    model = art.TrainableModel(
        name="email-agent-001",
        project="email-search-agent",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    # T4 友好设置
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(max_seq_length=8192),
        engine_args=art.dev.EngineArgs(enforce_eager=True, gpu_memory_utilization=0.8),
    )
    backend = LocalBackend(in_process=True, path="./.art")
    await model.register(backend)
    return model


async def run_training(
    max_steps: int = 20,
    groups_per_step: int = 2,
    rollouts_per_group: int = 4,
    num_epochs: int = 20,
    learning_rate: float = 1e-5,
):
    """训练主循环（完全异步、可参数化）"""
    if os.getenv("WANDB_API_KEY", ""):
        weave.init("email-search-agent", settings={"print_call_link": False})

    model = await setup_model_and_backend()
    train_scenarios = load_training_scenarios(split="train", limit=50, max_messages=1, shuffle=True, seed=42)

    step0 = await model.get_step()
    it = iterate_dataset(
        train_scenarios,
        groups_per_step=groups_per_step,
        num_epochs=num_epochs,
        initial_step=step0,
    )

    for batch in it:
        print(f"\nTraining step {batch.step} | epoch {batch.epoch} | epoch step {batch.epoch_step}")
        print(f"Batch scenarios: {len(batch.items)}")

        groups = []
        for scenario in batch.items:
            groups.append(
                art.TrajectoryGroup(
                    (rollout(model, EmailScenario(step=batch.step, scenario=scenario))
                     for _ in range(rollouts_per_group))
                )
            )

        finished = await art.gather_trajectory_groups(
            groups,
            pbar_desc="gather",
            max_exceptions=rollouts_per_group * len(batch.items),
        )
        extra_litellm_params = {"api_base": "http://localhost:6688", "api_key": os.environ["OPENAI_API_KEY"]}
        judged_groups = []
        for g in finished:
            judged = await ruler_score_group(group=g, judge_model="o4-mini", extra_litellm_params=extra_litellm_params,debug=True)
            judged_groups.append(judged)

        await model.delete_checkpoints()
        await model.train(
            judged_groups,
            config=art.TrainConfig(learning_rate=learning_rate),
            _config={"logprob_calculation_chunk_size": 8},
        )

        print(f"Completed training step {batch.step}")
        if batch.step >= max_steps:
            break

    print("\n✅ Training finished.")


# ========== 命令行入口 ==========

def ensure_env():
    """检查必要环境变量并加载 .env"""
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for RULER/judge (e.g., openai/o4-mini).")
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set -> skip W&B/Weave logging.")


def main():
    parser = argparse.ArgumentParser(description="Train the ART email search agent.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-db", help="Build SQLite DB (FTS) from Enron dataset.")
    p_train = sub.add_parser("train", help="Run training loop.")
    p_train.add_argument("--max-steps", type=int, default=20)
    p_train.add_argument("--groups-per-step", type=int, default=2)
    p_train.add_argument("--rollouts-per-group", type=int, default=4)
    p_train.add_argument("--num-epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()
    ensure_env()

    if args.cmd == "build-db":
        create_email_database()
        return

    if args.cmd == "train":
        asyncio.run(
            run_training(
                max_steps=args.max_steps,
                groups_per_step=args.groups_per_step,
                rollouts_per_group=args.rollouts_per_group,
                num_epochs=args.num_epochs,
                learning_rate=args.lr,
            )
        )


if __name__ == "__main__":
    main()
