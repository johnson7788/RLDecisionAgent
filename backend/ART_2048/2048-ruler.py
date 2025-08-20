#!/usr/bin/env python
# coding: utf-8
"""
训练 Qwen2.5-3B 模型玩 2048，使用 RULER 作为奖励机制。
judged_group， 使用 openai/o3 模型作为“裁判工具”。
它不会玩游戏，而是读取多条 trajectory（模型玩 2048 的过程），对它们做相对评分。
评分结果（reward）再反馈回训练循环。

Trajectory(
  messages_and_choices=[
    {"role": "system", "content": "You are an excellent 2048 player..."},
    {"role": "user", "content": "_ | 2 | _ | 4\n_ | _ | _ | _\n..."},
    {"role": "assistant", "content": "<move>left</move>"},
  ],
  metadata={"game_id": "ABC123", "step": 1},
  metrics={"max_value": 8, "board_value": 20, "move_number": 5},
  reward=0.65
)


流程：
1. 环境依赖与变量配置
2. 定义 2048 游戏环境
3. 定义可训练模型
4. Rollout（生成对局轨迹）
5. RULER 打分机制
6. 训练循环（带相对奖励）
7. 使用模型进行推理
"""

# ==============================
# 1. 环境依赖 & 环境变量
# ==============================
import os
import numpy as np

# 确保 numpy 版本 < 2.0.0（避免兼容性问题）
if (np.__version__).startswith("1."):
    print("Numpy version is 1.*.*, OK!")
else:
    raise ValueError("请重启运行环境，并确保 numpy<2.0.0")

# 必需：OpenAI API Key (RULER 使用 openai/o3 评估)
OPENAI_API_KEY = ""
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    raise ValueError("OPENAI_API_KEY 未设置，RULER 评估需要它。")

# 可选：W&B 日志
WANDB_API_KEY = ""
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
else:
    print("WANDB_API_KEY 未设置，将跳过日志记录。")

# ==============================
# 2. 定义 2048 游戏环境
# ==============================
import random
import string
import xml.etree.ElementTree as ET
from typing import Literal, TypedDict

from dotenv import load_dotenv
load_dotenv()

WINNING_VALUE = 128   # 胜利目标降低到 128，适合 T4 显存

class TwentyFortyEightGame(TypedDict):
    id: str
    board: list[list[int | None]]

def populate_random_cell(game: TwentyFortyEightGame) -> None:
    """随机在空格填 2 或 4"""
    coords = [(i, j) for i in range(4) for j in range(4) if game["board"][i][j] is None]
    i, j = random.choice(coords)
    game["board"][i][j] = 2 if random.random() < 0.9 else 4

def generate_game(board_length: int = 4) -> TwentyFortyEightGame:
    """生成新棋局"""
    game = {"id": "".join(random.choices(string.ascii_letters + string.digits, k=6)),
            "board": [[None]*board_length for _ in range(board_length)]}
    populate_random_cell(game); populate_random_cell(game)
    return game

def render_board(game: TwentyFortyEightGame) -> str:
    """渲染棋盘为字符串"""
    max_cell_width = max([len(str(cell)) for row in game["board"] for cell in row if cell])
    return "\n".join("|".join(
        str(cell).rjust(max_cell_width) if cell else "_".rjust(max_cell_width)
        for cell in row) for row in game["board"])

def condense_sequence(seq: list[int | None]) -> list[int | None]:
    """合并一行/列"""
    res, nums, i = [], [x for x in seq if x], 0
    while i < len(nums):
        if i+1 < len(nums) and nums[i] == nums[i+1]:
            res.append(nums[i]*2); i += 2
        else:
            res.append(nums[i]); i += 1
    return res + [None]*(4-len(res))

def condense_board(game: TwentyFortyEightGame, direction: Literal["left","right","up","down"]) -> None:
    """执行方向合并"""
    if direction == "left":
        for r in game["board"]: r[:] = condense_sequence(r)
    if direction == "right":
        for r in game["board"]: r[:] = condense_sequence(r[::-1])[::-1]
    if direction == "up":
        for c in range(4):
            col = [row[c] for row in game["board"]]
            for r,v in enumerate(condense_sequence(col)): game["board"][r][c]=v
    if direction == "down":
        for c in range(4):
            col = [row[c] for row in game["board"]]
            for r,v in enumerate(condense_sequence(col[::-1])[::-1]): game["board"][r][c]=v

def apply_agent_move(game: TwentyFortyEightGame, move_xml: str) -> None:
    """执行模型动作"""
    try:
        direction = ET.fromstring(move_xml).text
    except: raise ValueError("Invalid xml")
    if direction not in ["left","right","up","down"]: raise ValueError("Invalid direction")
    condense_board(game, direction); populate_random_cell(game)

def max_cell_value(game: TwentyFortyEightGame) -> int:
    return max(cell for row in game["board"] for cell in row if cell)

def check_game_finished(game: TwentyFortyEightGame) -> bool:
    if max_cell_value(game) >= WINNING_VALUE: return True
    if any(cell is None for row in game["board"] for cell in row): return False
    return True

def total_board_value(game: TwentyFortyEightGame) -> int:
    return sum(cell for row in game["board"] for cell in row if cell)


# ==============================
# 3. 定义可训练模型
# ==============================
import art
from art.local import LocalBackend
random.seed(42)

model = art.TrainableModel(
    name="agent-002",
    project="2048-multi-turn",
    base_model="Qwen/Qwen2.5-3B-Instruct",
)
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(max_seq_length=8192),
    engine_args=art.dev.EngineArgs(enforce_eager=True, gpu_memory_utilization=0.8),
)
backend = LocalBackend(in_process=True, path="./.art")
await model.register(backend)


# ==============================
# 4. Rollout 定义
# ==============================
import openai, requests, weave
from pydantic import BaseModel

if os.getenv("WANDB_API_KEY", ""):
    weave.init(model.project, settings={"print_call_link": False})

class Scenario2048(BaseModel): step: int

@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, scenario: Scenario2048) -> art.Trajectory:
    """执行一局游戏，返回 trajectory"""
    client = openai.AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)
    game, move_number = generate_game(), 0
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system",
          "content": "You are an excellent 2048 player. "
                     "Moves: left/right/up/down, return as <move>left</move>."}],
        metadata={"game_id": game["id"], "step": scenario.step}, reward=0,
    )
    while True:
        trajectory.messages_and_choices.append({"role": "user","content": render_board(game)})
        try:
            chat = await client.chat.completions.create(
                max_completion_tokens=128, messages=trajectory.messages(),
                model=model.name, stream=False)
        except Exception as e:
            print("chat completion failed", e); raise e
        choice, content = chat.choices[0], chat.choices[0].message.content
        trajectory.messages_and_choices.append(choice)
        try:
            apply_agent_move(game, content); move_number += 1
        except ValueError: break
        if check_game_finished(game):
            trajectory.metrics.update({"max_value": max_cell_value(game),
                                       "board_value": total_board_value(game),
                                       "move_number": move_number})
            break
    return trajectory


# ==============================
# 5. RULER 打分机制
# ==============================
from art.rewards import ruler_score_group

# 简单示例：三个不同表现的 trajectory
base_msgs = [{"role": "system", "content": "You count numbers."},
             {"role": "user", "content": "Count to 10."}]
good = art.Trajectory(messages_and_choices=[*base_msgs,
        {"role":"assistant","content":"1,2,3,4,5,6,7,8,9,10"}], reward=0)
mid  = art.Trajectory(messages_and_choices=[*base_msgs,
        {"role":"assistant","content":"1,2,3,4,5..."}], reward=0)
bad  = art.Trajectory(messages_and_choices=[*base_msgs,
        {"role":"assistant","content":"a,b,c"}], reward=0)

sample_group = art.TrajectoryGroup(trajectories=[good,mid,bad])
judged_group = await ruler_score_group(sample_group, "openai/o3", debug=True)

# 打印排名
for rank, traj in enumerate(sorted(judged_group.trajectories, key=lambda t: t.reward, reverse=True),1):
    print(f"\nRank {rank}: Score {traj.reward:.3f}")
    print(f"Response: {traj.messages()[-1]['content'][:50]}...")


# ==============================
# 6. 训练循环（带 RULER）
# ==============================
for i in range(await model.get_step(), 10):
    # 生成 trajectory groups
    train_groups = await art.gather_trajectory_groups(
        (art.TrajectoryGroup(rollout(model, Scenario2048(step=i)) for _ in range(18))),
        pbar_desc="gather", max_exceptions=18,
    )
    judged_groups = []
    for g in train_groups:
        judged_groups.append(await ruler_score_group(g, "openai/o3", debug=True))
    await model.delete_checkpoints()
    await model.train(
        judged_groups, config=art.TrainConfig(learning_rate=1e-5),
        _config={"logprob_calculation_chunk_size": 8},
    )


# ==============================
# 7. 使用模型推理
# ==============================
import torch
from unsloth import FastLanguageModel

lora_model_path = f".art/{model.project}/models/{model.name}/{await model.get_step():04d}"
print(f"loading model from {lora_model_path}")

peft_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_model_path, max_seq_length=16384,
    dtype=torch.bfloat16, load_in_4bit=True,
)
FastLanguageModel.for_inference(peft_model)

game, move_number = generate_game(), 0
messages = [{"role": "system",
             "content": "You are an excellent 2048 player. Moves as <move>...</move>."}]

while not check_game_finished(game):
    messages.append({"role":"user","content": render_board(game)})
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    with torch.no_grad():
        outputs = peft_model.generate(input_ids=inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)
    content = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    messages.append({"role":"assistant","content": content})
    try: apply_agent_move(game, content); move_number += 1
    except ValueError: break
    if move_number % 10 == 0:
        print(f"\nmove {move_number}\nboard:\n{render_board(game)}\nmove: {content}")

print(f"\nGame finished in {move_number} moves")
print(f"Max: {max_cell_value(game)}, Board Sum: {total_board_value(game)}")
