#!/usr/bin/env python
# coding: utf-8
"""
训练 Qwen2.5-3B 模型玩 2048 游戏的示例。
主要流程：
1. 定义 2048 游戏环境
2. 定义可训练模型 (LoRA 微调)
3. 定义 rollout 函数（模型与环境交互）
4. 训练循环
"""

# ==============================
# 1. 环境依赖 & 配置
# ==============================
import os
import random
import string
import math
import xml.etree.ElementTree as ET
from typing import Literal, TypedDict

from dotenv import load_dotenv
import requests
import weave
from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from art.local import LocalBackend

load_dotenv()

# 设置目标：避免显存不足，将胜利条件从 2048 降到 128
WINNING_VALUE = 128


# ==============================
# 2. 定义 2048 游戏环境
# ==============================
class TwentyFortyEightGame(TypedDict):
    """存储游戏状态"""
    id: str
    board: list[list[int | None]]


def populate_random_cell(game: TwentyFortyEightGame) -> None:
    """随机生成 2 或 4 填充到空格子"""
    all_clear_coordinates = [
        (i, j)
        for i in range(len(game["board"]))
        for j in range(len(game["board"][i]))
        if game["board"][i][j] is None
    ]
    random_clear_coordinates = random.choice(all_clear_coordinates)
    game["board"][random_clear_coordinates[0]][random_clear_coordinates[1]] = (
        2 if random.random() < 0.9 else 4
    )


def generate_game(board_length: int = 4) -> TwentyFortyEightGame:
    """生成新游戏，棋盘初始化并放置两个数字"""
    id = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    game = {
        "id": id,
        "board": [[None for _ in range(board_length)] for _ in range(board_length)],
    }
    populate_random_cell(game)
    populate_random_cell(game)
    return game


def render_board(game: TwentyFortyEightGame) -> str:
    """渲染棋盘，返回可读字符串"""
    board = game["board"]
    max_cell_width = max(
        [len(str(cell)) for row in board for cell in row if cell is not None]
    )
    board_str = ""
    for row in board:
        board_str += "|".join(
            [
                str(cell).rjust(max_cell_width)
                if cell is not None
                else "_".rjust(max_cell_width)
                for cell in row
            ]
        )
        board_str += "\n"
    return board_str


def condense_sequence(sequence: list[int | None]) -> list[int | None]:
    """合并一行或一列的数字（2048 核心规则）"""
    condensed_sequence = []
    gapless_sequence = [cell for cell in sequence if cell is not None]

    i = 0
    while i < len(gapless_sequence):
        if (
            i + 1 < len(gapless_sequence)
            and gapless_sequence[i] == gapless_sequence[i + 1]
        ):
            condensed_sequence.append(gapless_sequence[i] * 2)
            i += 2
        else:
            condensed_sequence.append(gapless_sequence[i])
            i += 1
    return condensed_sequence + [None] * (4 - len(condensed_sequence))


def condense_board(game: TwentyFortyEightGame, direction: Literal["left", "right", "up", "down"]) -> None:
    """根据方向合并棋盘"""
    if direction == "left":
        for row in game["board"]:
            row[:] = condense_sequence(row)

    if direction == "right":
        for row in game["board"]:
            row[:] = condense_sequence(row[::-1])[::-1]

    if direction == "up":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            condensed_column = condense_sequence(column)
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]

    if direction == "down":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            condensed_column = condense_sequence(column[::-1])[::-1]
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]


def apply_agent_move(game: TwentyFortyEightGame, move_xml: str) -> None:
    """执行模型的动作（从 XML 解析方向）"""
    try:
        root = ET.fromstring(move_xml)
        direction = root.text
    except Exception:
        raise ValueError("Invalid xml")

    if direction not in ["left", "right", "up", "down"]:
        raise ValueError("Invalid direction")

    condense_board(game, direction)
    populate_random_cell(game)


def max_cell_value(game: TwentyFortyEightGame) -> int:
    """返回棋盘最大数值"""
    return max([cell for row in game["board"] for cell in row if cell is not None])


def check_game_finished(game: TwentyFortyEightGame) -> bool:
    """判断游戏是否结束"""
    if max_cell_value(game) >= WINNING_VALUE:
        return True
    if any(cell is None for row in game["board"] for cell in row):
        return False
    return True


def total_board_value(game: TwentyFortyEightGame) -> int:
    """返回棋盘所有数字之和"""
    return sum([cell for row in game["board"] for cell in row if cell is not None])


# ==============================
# 3. 定义可训练模型
# ==============================
random.seed(42)

model = art.TrainableModel(
    name="agent-002",
    project="2048-multi-turn",
    base_model="Qwen/Qwen2.5-3B-Instruct",
)

# 调整参数，适配 T4 GPU
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(max_seq_length=8192),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    ),
)

backend = LocalBackend(in_process=True, path="./.art")
await model.register(backend)


# ==============================
# 4. 定义 Rollout
# ==============================
class Scenario2048(BaseModel):
    step: int


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, scenario: Scenario2048) -> art.Trajectory:
    """运行一局游戏，生成 trajectory（对话历史 + 动作 + reward）"""
    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)
    game = generate_game()
    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system",
             "content": "你是一个 2048 高手。请输出 <move>方向</move>，方向可选 'left' 'right' 'up' 'down'。"}
        ],
        metadata={"game_id": game["id"], "notebook-id": "2048", "step": scenario.step},
        reward=0,
    )

    while True:
        trajectory.messages_and_choices.append({"role": "user", "content": render_board(game)})

        # 调用模型生成动作
        messages = trajectory.messages()
        chat_completion = await client.chat.completions.create(
            max_completion_tokens=128,
            messages=messages,
            model=model.name,
            stream=False,
        )

        choice = chat_completion.choices[0]
        content = choice.message.content
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
            move_number += 1
        except ValueError:  # 非法动作，奖励 -1
            trajectory.reward = -1
            break

        if check_game_finished(game):
            max_value = max_cell_value(game)
            board_value = total_board_value(game)
            trajectory.metrics.update({
                "max_value": max_value,
                "board_value": board_value,
                "move_number": move_number,
            })

            if max_value < WINNING_VALUE:
                # 奖励函数：按 log 缩放
                max_value_reward = (math.log(max_value, 2) - 1) / (math.log(WINNING_VALUE, 2) - 1)
                board_value_reward = (math.log(board_value, 2) - 1) / (math.log(WINNING_VALUE * 16, 2) - 1)
                trajectory.reward = max_value_reward + (board_value_reward * 0.2)
            else:
                trajectory.reward = 2  # 胜利奖励加倍
            break

    return trajectory


# ==============================
# 5. 训练循环
# ==============================
for i in range(await model.get_step(), 10):
    # 每一步并行生成 18 个 trajectory group
    train_groups = await art.gather_trajectory_groups(
        (art.TrajectoryGroup(rollout(model, Scenario2048(step=i)) for _ in range(18))),
        pbar_desc="gather",
        max_exceptions=18,
    )

    # 删除旧 checkpoint，仅保留最新
    await model.delete_checkpoints()

    # 训练模型
    await model.train(
        train_groups,
        config=art.TrainConfig(learning_rate=1e-5),
        _config={"logprob_calculation_chunk_size": 8},  # 节省显存
    )
