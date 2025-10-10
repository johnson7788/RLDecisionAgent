#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/13 20:36
# @File  : my_ruler.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 自定义的ruler函数，兼容Deepseek，只需要在.env中添加以下内容：
# 评估模型的配置

"""
RULER (Relative Universal LLM-Elicited Rewards) - A general-purpose reward function for RL agents.
"""

import json
import re
import logging
from textwrap import dedent
from typing import List
from rich import print
from rich.logging import RichHandler
from litellm import acompletion
from litellm.types.utils import ModelResponse
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field, ValidationError
import asyncio
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Awaitable, Iterable, Iterator, cast, overload
from typing import Literal

import pydantic
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion import Choice

Message = ChatCompletionMessageParam
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
Tools = list[ChatCompletionToolParam]
MetadataValue = float | int | str | bool | None

def get_messages(messages_and_choices: MessagesAndChoices) -> Messages:
    messages: Messages = []
    for message_or_choice in messages_and_choices:
        if isinstance(message_or_choice, Choice):
            content = message_or_choice.message.content or ""
            tool_calls = message_or_choice.message.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": tool_call.type,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                for tool_call in tool_calls
                            ]
                        }
                        if tool_calls
                        else {}
                    ),  # type: ignore
                }
            )
        else:
            # Ensure content is always a string for tokenizer chat templates
            msg = dict(message_or_choice)
            if msg.get("content") is None:
                msg["content"] = ""
            messages.append(msg)  # type: ignore[arg-type]
    return messages

class PydanticException(pydantic.BaseModel):
    type: str
    message: str
    traceback: str



class History(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    tools: Tools | None = None

    def messages(self) -> Messages:
        return get_messages(self.messages_and_choices)



class Trajectory(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    tools: Tools | None = None
    additional_histories: list[History] = []
    reward: float
    metrics: dict[str, float | int | bool] = {}
    metadata: dict[str, MetadataValue] = {}
    logs: list[str] = []
    start_time: datetime = pydantic.Field(default_factory=datetime.now, exclude=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.start_time = datetime.now()

    def log(self, message: str) -> None:
        self.logs.append(message)

    def finish(self) -> "Trajectory":
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics["duration"] = duration
        return self

    @asynccontextmanager
    async def track_duration(self, metric_name: str) -> AsyncGenerator[None, None]:
        start_time = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            metric_key = f"{metric_name}_duration"
            self.metrics[metric_key] = self.metrics.get(metric_key, 0.0) + duration

    def __str__(self) -> str:
        return f"Trajectory(reward={self.reward}, metrics={self.metrics}, metadata={self.metadata})"

    def messages(self) -> Messages:
        return get_messages(self.messages_and_choices)

    # Used for logging to console
    def for_logging(self) -> dict[str, Any]:
        loggable_dict = {
            "reward": self.reward,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "messages": [],
            "tools": self.tools,
            "logs": self.logs,
        }
        for message_or_choice in self.messages_and_choices:
            trainable = isinstance(message_or_choice, Choice)
            message = (
                message_or_choice.message.to_dict() if trainable else message_or_choice
            )
            loggable_dict["messages"].append({**message, "trainable": trainable})
        return loggable_dict


class TrajectoryGroup(pydantic.BaseModel):
    trajectories: list[Trajectory]
    exceptions: list[PydanticException] = []

    def __init__(
        self,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ),
        *,
        exceptions: list[BaseException] = [],
    ) -> None:
        super().__init__(
            trajectories=[
                trajectory
                for trajectory in trajectories
                if isinstance(trajectory, Trajectory)
            ]
            or getattr(self, "trajectories", []),
            exceptions=[
                PydanticException(
                    type=str(type(exception)),
                    message=str(exception),
                    traceback="\n".join(
                        traceback.format_exception(
                            type(exception), exception, exception.__traceback__
                        )
                    ),
                )
                for exception in (
                    [
                        exception
                        for exception in trajectories
                        if isinstance(exception, BaseException)
                    ]
                    + exceptions
                )
            ],
        )

    def __iter__(self) -> Iterator[Trajectory]:  # type: ignore[override]
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Trajectory | BaseException],
        *,
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup": ...

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Awaitable[Trajectory]],
        *,
        exceptions: list[BaseException] = [],
    ) -> Awaitable["TrajectoryGroup"]: ...

    def __new__(
        cls,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ),
        *,
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup | Awaitable[TrajectoryGroup]":
        ts = list(trajectories)
        if any(hasattr(t, "__await__") for t in ts):

            async def _(exceptions: list[BaseException]):
                from .gather import get_gather_context, record_metrics

                context = get_gather_context()
                trajectories = []
                for future in asyncio.as_completed(
                    cast(list[Awaitable[Trajectory]], ts)
                ):
                    try:
                        trajectory = await future
                        trajectories.append(trajectory)
                        record_metrics(context, trajectory)
                        context.update_pbar(n=1)
                    except BaseException as e:
                        exceptions.append(e)
                        context.metric_sums["exceptions"] += 1
                        context.update_pbar(n=0)
                        if context.too_many_exceptions():
                            raise
                return TrajectoryGroup(
                    trajectories=trajectories,
                    exceptions=exceptions,
                )

            class CoroutineWithMetadata:
                def __init__(self, coro, num_trajectories):
                    self.coro = coro
                    self._num_trajectories = num_trajectories

                def __await__(self):
                    return self.coro.__await__()

            coro = _(exceptions.copy())
            return CoroutineWithMetadata(coro, len(ts))
        else:
            group = super().__new__(cls)
            group.__init__(
                trajectories=cast(list[Trajectory | BaseException], ts),
                exceptions=exceptions,
            )
            return group
# ====== 日志：中文输出，默认 INFO；首次导入时仅在无处理器时配置 ======
logger = logging.getLogger("RULER")
if not logger.handlers:
    try:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False, markup=True)
    except Exception:
        handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

# 如需观察 litellm 的 HTTP 往返，可保留这一行（会非常啰嗦）
# litellm._turn_on_debug()

class TrajectoryScore(BaseModel):
    """Individual score for a single trajectory."""
    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(description="A short description of the trajectory's performance.")
    score: float = Field(ge=0.0, le=1.0, description="A score between 0 and 1.")


class Response(BaseModel):
    """Response format expected from the LLM judge."""
    scores: List[TrajectoryScore] = Field(description="The scores for each trajectory.")


DEFAULT_RUBRIC = dedent(
    """         
        - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
        - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
        - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
        - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.
    """
)

# 当模型不支持 structured output（如 deepseek）时，用于强约束 JSON 输出的追加说明
JSON_ONLY_INSTRUCTIONS = dedent(
    """
    Output format:
    - You MUST reply with ONLY a single JSON object.
    - Use standard JSON: double quotes for all strings; no trailing commas; no comments; no additional text.
    - Shape:
      {
        "scores": [
          {"trajectory_id": "1", "explanation": "<short reason>", "score": 0.0},
          {"trajectory_id": "2", "explanation": "<short reason>", "score": 0.0}
          // ... one item per trajectory, ids are "1", "2", ...
        ]
      }
    """
)

def _supports_response_format(model_name: str) -> bool:
    """简单判定是否支持 OpenAI-style response_format。DeepSeek 家族通常不支持。"""
    name = (model_name or "").lower()
    return not ("deepseek" in name)

def _extract_first_json_object(text: str) -> str | None:
    """从模型输出中提取第一个平衡的 JSON 对象；兼容 ```json ... ``` 代码块或混入文本"""
    if not text:
        return None
    codeblock = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if codeblock:
        logger.debug("从代码块中提取到 JSON。")
        return codeblock.group(1).strip()

    start_idx = None
    brace_depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start_idx is None:
                start_idx = i
            brace_depth += 1
        elif ch == "}":
            if start_idx is not None:
                brace_depth -= 1
                if brace_depth == 0:
                    candidate = text[start_idx : i + 1]
                    logger.debug("通过大括号平衡扫描提取到 JSON。")
                    return candidate.strip()
    logger.debug("未能从文本中提取到 JSON。")
    return None

def _parse_response_to_model(content: str | dict) -> Response:
    """尽力把字符串/字典解析成 Response；包含若干兜底路径"""
    if isinstance(content, dict):
        logger.debug("输入已是字典对象，直接进行模型校验。")
        return Response.model_validate(content)

    try:
        logger.debug("尝试直接解析完整字符串为 JSON 并校验。")
        return Response.model_validate_json(content)
    except Exception:
        logger.debug("直接解析失败，尝试提取首个 JSON 对象。")

    extracted = _extract_first_json_object(content or "")
    if extracted:
        try:
            logger.debug("尝试对提取出的 JSON 做严格解析并校验。")
            obj = json.loads(extracted)
            return Response.model_validate(obj)
        except Exception:
            logger.debug("严格解析失败，尝试保守替换引号后再解析。")
            try:
                cleaned = extracted
                if "'" in cleaned and '"' not in cleaned:
                    cleaned = cleaned.replace("'", '"')
                obj = json.loads(cleaned)
                return Response.model_validate(obj)
            except Exception as e2:
                raise ValueError(f"Failed to parse JSON after extraction: {e2}") from e2

    raise ValueError("Unable to parse model content into Response JSON.")

async def ruler(
    message_lists: list[list[ChatCompletionMessageParam]],
    judge_model: str = "openai/o3",
    extra_litellm_params: dict | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
) -> list[TrajectoryScore]:
    """核心 RULER：对多条轨迹进行评分并返回得分列表。"""
    # 动态调整日志级别
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("已启用 DEBUG 日志。")

    if not message_lists:
        logger.warning("输入的 message_lists 为空，直接返回空结果。")
        return []

    logger.info("开始执行 RULER 评分：共 [bold]%d[/] 条轨迹，评审模型：%s", len(message_lists), judge_model)

    # 计算公共前缀（节省 token）
    common_prefix_len = 0
    for idx, msg in enumerate(message_lists[0]):
        if all(len(msg_list) > idx and msg_list[idx] == msg for msg_list in message_lists):
            common_prefix_len += 1
        else:
            break
    logger.debug("计算得到的公共前缀长度：%d", common_prefix_len)

    user_text = ""
    if common_prefix_len > 0:
        common_prefix_messages = message_lists[0][:common_prefix_len]
        user_text += "<context>\n" + json.dumps(common_prefix_messages, ensure_ascii=False) + "\n</context>\n\n"
        logger.debug("已序列化公共前缀上下文。")

    serialized_trajectories: List[str] = []
    for idx, full_messages in enumerate(message_lists, start=1):
        trimmed_messages = full_messages[common_prefix_len:]
        serialized_trajectories.append(
            f'<trajectory id="{idx}">\n' + json.dumps(trimmed_messages, ensure_ascii=False) + "\n</trajectory>"
        )
    user_text += "Trajectories:\n\n" + "\n\n".join(serialized_trajectories)
    logger.debug("已序列化全部轨迹。")

    judge_prompt = dedent(
        f"""
        All of the trajectories below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.

        Grading standards:
        {rubric}
        """
    )

    supports_structured = _supports_response_format(judge_model)
    logger.info("模型是否支持结构化输出（response_format）：%s", "是" if supports_structured else "否")

    if not supports_structured:
        judge_prompt = judge_prompt + "\n" + JSON_ONLY_INSTRUCTIONS
        logger.debug("已为不支持结构化输出的模型追加严格 JSON 指令。")

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_text},
    ]
    logger.debug(f"已构造消息列表（system + user）: {messages}")

    # 构造 acompletion 的参数
    litellm_kwargs = dict(extra_litellm_params or {})
    if supports_structured:
        litellm_kwargs["response_format"] = Response
        logger.debug("已设置 response_format 为 Pydantic 模型。")

    max_retries = 3 if not supports_structured else 1
    parsed: Response | None = None
    last_raw_content: str | dict | None = None
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        retry_messages = list(messages)
        if attempt > 1 and not supports_structured:
            retry_messages = retry_messages + [{
                "role": "system",
                "content": ("Reminder: Return ONLY a single strict JSON object matching the schema. "
                            "No code fences, no extra text, no trailing commas. If previous reply was invalid, fix it now.")
            }]
            if "temperature" not in litellm_kwargs:
                litellm_kwargs["temperature"] = 0
            logger.debug("第 %d 次重试：已追加更严格的 JSON 约束，并将 temperature 设置为 0。", attempt)

        try:
            logger.debug("第 %d/%d 次请求评审模型……", attempt, max_retries)
            response = await acompletion(
                model=judge_model,
                messages=retry_messages,
                caching=False,
                **litellm_kwargs,
            )
            assert isinstance(response, ModelResponse)
            if len(response.choices) == 0:
                raise ValueError(f"No choices in response: {response}")

            first_choice = response.choices[0]
            raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
            last_raw_content = raw_content

            if debug:
                try:
                    preview = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                    logger.debug("模型原始输出（预览 JSON）：%s", preview)
                except json.JSONDecodeError:
                    # 直接打印字符串
                    logger.debug("模型原始输出（非 JSON 文本预览，前 300 字）：%s", str(raw_content)[:300])

            parsed = _parse_response_to_model(raw_content)
            logger.info("第 %d 次尝试解析成功，得到 %d 条评分结果。", attempt, len(parsed.scores))
            break

        except (ValidationError, ValueError, AssertionError) as e:
            last_error = e
            logger.warning("第 %d 次尝试解析失败：%s", attempt, e)
            continue

    if parsed is None:
        if supports_structured:
            logger.error("解析失败：支持结构化输出的模型在 %d 次尝试后仍无法解析。", max_retries)
            raise ValueError(
                f"Failed to parse/validate judge response after {max_retries} attempt(s). "
                f"Last error: {last_error}\nRaw content: {last_raw_content}"
            )
        else:
            logger.error(
                "连续 %d 次解析失败，针对不支持结构化输出的模型，回退为全 0 分。原始内容片段：%s",
                max_retries, str(last_raw_content)[:500]
            )
            zero_scores: List[TrajectoryScore] = [
                TrajectoryScore(
                    trajectory_id=str(i + 1),
                    explanation="(fallback) JSON 解析连续失败，已回退为 0 分",
                    score=0.0,
                )
                for i in range(len(message_lists))
            ]
            return zero_scores

    # 一致性检查与补齐/截断
    if len(parsed.scores) != len(message_lists):
        logger.warning(
            "返回的评分数量(%d)与轨迹数量(%d)不一致，正在进行补齐/截断以保持一致性。",
            len(parsed.scores), len(message_lists)
        )
        fixed_scores: List[TrajectoryScore] = []
        for i in range(len(message_lists)):
            if i < len(parsed.scores):
                s = parsed.scores[i]
            else:
                s = TrajectoryScore(trajectory_id=str(i + 1), explanation="(缺失，已按顺序补齐)", score=0.0)
            s.trajectory_id = str(i + 1)
            fixed_scores.append(s)
        parsed.scores = fixed_scores

    logger.info("RULER 评分完成。")
    return parsed.scores


async def ruler_score_group(
    group: TrajectoryGroup,
    judge_model: str = "openai/o3",
    extra_litellm_params: dict | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    swallow_exceptions: bool = False,
    debug: bool = True,
) -> TrajectoryGroup | None:
    """使用 RULER 对 TrajectoryGroup 打分，便于训练循环使用。"""
    # 动态调整日志级别
    logger.info("开始对 TrajectoryGroup 进行 RULER 打分，共 %d 条轨迹。", len(group.trajectories))

    for traj in group.trajectories:
        if len(traj.additional_histories) > 0:
            logger.error("检测到 additional_histories 非空，当前 RULER 尚不支持。")
            raise ValueError("Additional histories are not supported by RULER yet.")

    new_trajectories = []
    for t in group.trajectories:
        new_traj = t.__class__(
            messages_and_choices=t.messages_and_choices.copy(),
            tools=t.tools.copy() if t.tools else None,
            additional_histories=[h.model_copy(deep=True) for h in t.additional_histories],
            reward=t.reward,
            metrics=t.metrics.copy(),
            metadata=t.metadata.copy(),
            logs=t.logs.copy(),
        )
        new_trajectories.append(new_traj)
    logger.debug("已拷贝 %d 条轨迹为新对象，准备构造 message_lists。", len(new_trajectories))

    message_lists: list[list[ChatCompletionMessageParam]] = []
    for traj in new_trajectories:
        message_lists.append(traj.messages())
        traj.metrics["independent_reward"] = traj.reward
    logger.debug("已构造完 message_lists 并同步 independent_reward。")

    try:
        logger.info("调用 ruler() 开始评分……")
        scores = await ruler(
            message_lists,
            judge_model=judge_model,
            extra_litellm_params=extra_litellm_params,
            rubric=rubric,
            debug=debug,
        )
    except Exception as e:
        if swallow_exceptions:
            logger.warning("[art_ruler] 捕获异常并吞掉：%s", e)
            print(f"[art_ruler] Swallowed exception: {e}")
            return None
        else:
            logger.error("评分过程中发生异常：%s", e)
            raise

    for traj, score in zip(new_trajectories, scores):
        print(f"traj {traj} 的得分是: {score}")
        traj.metrics["ruler_score"] = score.score
        traj.reward = score.score
        traj.log(f"RULER explanation: {score.explanation}")
    logger.info("已将得分与解释写回 TrajectoryGroup。")

    return TrajectoryGroup(new_trajectories)
