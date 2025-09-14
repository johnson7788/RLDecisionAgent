#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/13 20:36
# @File  : my_ruler.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :


"""
RULER (Relative Universal LLM-Elicited Rewards) - A general-purpose reward function for RL agents.
"""

import json
import re  # NEW
from textwrap import dedent
from typing import List

from litellm import acompletion
from litellm.types.utils import ModelResponse
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field, ValidationError  # CHANGED
from rich import print

import art


class TrajectoryScore(BaseModel):
    """Individual score for a single trajectory."""
    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(description="A short description of the trajectory's performance.")
    score: float = Field(ge=0.0, le=1.0, description="A score between 0 and 1.")  # NEW: explicit bounds


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

# NEW: 当模型不支持 structured output（如 deepseek）时，用于强约束 JSON 输出的追加说明
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

# NEW: 简单判定是否支持 response_format
def _supports_response_format(model_name: str) -> bool:
    name = (model_name or "").lower()
    # DeepSeek 家族目前不支持 OpenAI-style response_format
    return not ("deepseek" in name)

# NEW: 从模型输出中提取第一个平衡的 JSON 对象；兼容 ```json ... ``` 代码块或混入文本
def _extract_first_json_object(text: str) -> str | None:
    if not text:
        return None
    # 优先从 ```json 代码块``` 中抓取
    codeblock = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if codeblock:
        return codeblock.group(1).strip()

    # 退而求其次：扫描第一个平衡的大括号对象
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
                    return candidate.strip()
    return None

# NEW: 统一的解析入口——尽力把字符串/字典解析成 Response
def _parse_response_to_model(content: str | dict) -> Response:
    # 有些提供商可能直接返回已解析的对象
    if isinstance(content, dict):
        return Response.model_validate(content)

    # 标准路径：直接解析完整字符串
    try:
        return Response.model_validate_json(content)
    except Exception:
        pass

    # 提取第一个 JSON 对象再试
    extracted = _extract_first_json_object(content or "")
    if extracted:
        # 再尝试严格 JSON 解析
        try:
            obj = json.loads(extracted)
            return Response.model_validate(obj)
        except Exception:
            # 兜底：非常保守地替换单引号为双引号（可能带来误判，但在 deepseek 偶发输出时有用）
            try:
                cleaned = extracted
                if "'" in cleaned and '"' not in cleaned:
                    cleaned = cleaned.replace("'", '"')
                obj = json.loads(cleaned)
                return Response.model_validate(obj)
            except Exception as e2:
                raise ValueError(f"Failed to parse JSON after extraction: {e2}") from e2

    # 全部失败
    raise ValueError("Unable to parse model content into Response JSON.")


async def ruler(
    message_lists: list[list[ChatCompletionMessageParam]],
    judge_model: str = "openai/o3",
    extra_litellm_params: dict | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
) -> list[TrajectoryScore]:
    """Core RULER implementation that scores a list of message trajectories."""
    if not message_lists:
        return []

    # 计算公共前缀（节省 token）
    common_prefix_len = 0
    for idx, msg in enumerate(message_lists[0]):
        if all(len(msg_list) > idx and msg_list[idx] == msg for msg_list in message_lists):
            common_prefix_len += 1
        else:
            break

    user_text = ""
    if common_prefix_len > 0:
        common_prefix_messages = message_lists[0][:common_prefix_len]
        user_text += "<context>\n" + json.dumps(common_prefix_messages) + "\n</context>\n\n"

    serialized_trajectories: List[str] = []
    for idx, full_messages in enumerate(message_lists, start=1):
        trimmed_messages = full_messages[common_prefix_len:]
        serialized_trajectories.append(
            f'<trajectory id="{idx}">\n' + json.dumps(trimmed_messages) + "\n</trajectory>"
        )

    user_text += "Trajectories:\n\n" + "\n\n".join(serialized_trajectories)

    judge_prompt = dedent(
        f"""
        All of the trajectories below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.

        Grading standards:
        {rubric}
        """
    )

    # NEW: 若模型不支持 response_format，则在系统提示中追加“只输出 JSON”的硬性约束
    supports_structured = _supports_response_format(judge_model)
    if not supports_structured:
        judge_prompt = judge_prompt + "\n" + JSON_ONLY_INSTRUCTIONS

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_text},
    ]

    # 构造 acompletion 的参数
    litellm_kwargs = dict(extra_litellm_params or {})
    # CHANGED: DeepSeek 等不传 response_format；其他模型仍可享受 pydantic 结构化输出
    if supports_structured:
        litellm_kwargs["response_format"] = Response

    response = await acompletion(
        model=judge_model,
        messages=messages,
        caching=False,
        **litellm_kwargs,
    )
    assert isinstance(response, ModelResponse)

    if len(response.choices) == 0:
        raise ValueError(f"No choices in response: {response}")
    first_choice = response.choices[0]

    raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
    if debug:
        try:
            parsed_preview = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
            print("\n[RULER] Pretty-printed LLM choice JSON (raw):")
            print(parsed_preview)
        except json.JSONDecodeError:
            print("\n[RULER] Raw choice content (non-JSON):")
            print(raw_content)

    # NEW: 统一解析，无论是否使用了 response_format
    try:
        parsed = _parse_response_to_model(raw_content)
    except (ValidationError, ValueError) as e:
        raise ValueError(f"Failed to parse/validate judge response: {e}\nRaw content: {raw_content}") from e

    # NEW: 如果模型没按要求返回完整列表，进行最小化一致性检查
    if len(parsed.scores) != len(message_lists):
        # 若数量不符，尝试按顺序补齐/截断（保守处理）
        fixed_scores: List[TrajectoryScore] = []
        for i in range(len(message_lists)):
            if i < len(parsed.scores):
                s = parsed.scores[i]
            else:
                s = TrajectoryScore(trajectory_id=str(i + 1), explanation="(missing, filled by fallback)", score=0.0)
            # 统一成 "1","2",...
            s.trajectory_id = str(i + 1)
            fixed_scores.append(s)
        parsed.scores = fixed_scores

    return parsed.scores


async def ruler_score_group(
    group: art.TrajectoryGroup,
    judge_model: str = "openai/o3",
    extra_litellm_params: dict | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    swallow_exceptions: bool = False,
    debug: bool = False,
) -> art.TrajectoryGroup | None:
    """Score a trajectory group using RULER for use in training loops."""
    for traj in group.trajectories:
        if len(traj.additional_histories) > 0:
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

    message_lists: list[list[ChatCompletionMessageParam]] = []
    for traj in new_trajectories:
        message_lists.append(traj.messages())
        traj.metrics["independent_reward"] = traj.reward

    try:
        scores = await ruler(
            message_lists,
            judge_model=judge_model,
            extra_litellm_params=extra_litellm_params,
            rubric=rubric,
            debug=debug,
        )
    except Exception as e:
        if swallow_exceptions:
            print(f"[art_ruler] Swallowed exception: {e}")
            return None
        else:
            raise

    for traj, score in zip(new_trajectories, scores):
        traj.metrics["ruler_score"] = score.score
        traj.reward = score.score
        traj.log(f"RULER explanation: {score.explanation}")

    return art.TrajectoryGroup(new_trajectories)
