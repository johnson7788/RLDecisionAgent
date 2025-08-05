#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc: 生成包含 SFT + RL 的训练和测试集（json + parquet）

import random
import json
import os
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split


def generate_guess_game_data(n=1000, num_range=(1, 100), max_turns=6):
    sft_data = []
    rl_data = []

    for _ in range(n):
        target = random.randint(*num_range)
        turns = []
        turns.append({
            "role": "system",
            "content": f"你能和用户一起玩猜数字，假如你现在心里想的数字是{target}，用户说任意一个数字，你只会说大了，小了，或者🎉Bingo正确"
        })

        low, high = num_range
        guessed_correctly = False
        assistant_responses = []

        for attempt in range(max_turns):
            guess = random.randint(low, high)
            turns.append({"role": "user", "content": str(guess)})
            if guess < target:
                response = "小了"
                low = guess + 1
            elif guess > target:
                response = "大了"
                high = guess - 1
            else:
                response = "🎉Bingo正确"
                guessed_correctly = True

            turns.append({"role": "assistant", "content": response})
            assistant_responses.append(response)

            if guessed_correctly:
                break

        if not guessed_correctly:
            final_response = f"没猜中哦，正确答案是 {target}"
            turns.append({"role": "assistant", "content": final_response})
            assistant_responses.append(final_response)

        # SFT 格式：全对话
        sft_data.append({"messages": deepcopy(turns)})

        # RL 格式，这里不对啊
        rl_data.append({
            "prompt": deepcopy(turns),
            "data_source": "guess_number",
            "ability": "other",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(target)
            },
            "extra_info": {
                "response": assistant_responses
            }
        })

    return sft_data, rl_data


def save_datasets(output_dir, sft_data, rl_data):
    os.makedirs(os.path.join(output_dir, "sft"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rl"), exist_ok=True)

    # === 保存 JSON 原始数据 ===
    with open(os.path.join(output_dir, "sft", "raw.json"), "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "rl", "raw.json"), "w", encoding="utf-8") as f:
        json.dump(rl_data, f, ensure_ascii=False, indent=2)

    # === 切分 SFT 数据 ===
    sft_train, sft_test = train_test_split(sft_data, test_size=0.1, random_state=42)
    sft_train_df = pd.DataFrame(sft_train)
    sft_test_df = pd.DataFrame(sft_test)
    sft_train_df.to_parquet(os.path.join(output_dir, "sft", "train.parquet"))
    sft_test_df.to_parquet(os.path.join(output_dir, "sft", "test.parquet"))

    # === 切分 RL 数据 ===
    rl_train, rl_test = train_test_split(rl_data, test_size=0.1, random_state=42)
    rl_train_df = pd.DataFrame(rl_train)
    rl_test_df = pd.DataFrame(rl_test)
    rl_train_df.to_parquet(os.path.join(output_dir, "rl", "train.parquet"))
    rl_test_df.to_parquet(os.path.join(output_dir, "rl", "test.parquet"))

    print(f"✅ SFT train: {len(sft_train)}, test: {len(sft_test)}")
    print(f"✅ RL  train: {len(rl_train)}, test: {len(rl_test)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./guess_number_dataset")
    args = parser.parse_args()

    sft_data, rl_data = generate_guess_game_data(n=args.num_samples)
    save_datasets(args.output_dir, sft_data, rl_data)
