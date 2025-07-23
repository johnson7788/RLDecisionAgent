#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/7/21 17:05
# @File  : create_dataset.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 创建数据
import os
import json
import random
import pandas as pd

prompt_template = 'Please convert the string "{}" to emojis.'

# 自定义一些翻译策略
char_to_emoji_map = {
    'a': '🍎', 'b': '🐝', 'c': '🐱', 'd': '🐶', 'e': '🐘',
    'f': '🐸', 'g': '🦒', 'h': '🕳️', 'i': '🍦', 'j': '🕹️',
    'k': '🎋', 'l': '🦁', 'm': '🌝', 'n': '👃', 'o': '🐙',
    'p': '🥞', 'q': '👸', 'r': '🤖', 's': '🐍', 't': '🌴',
    'u': '☂️', 'v': '🎻', 'w': '🌊', 'x': '❌', 'y': '🛳️', 'z': '⚡',
    '0': '0️⃣', '1': '1️⃣', '2': '2️⃣', '3': '3️⃣', '4': '4️⃣',
    '5': '5️⃣', '6': '6️⃣', '7': '7️⃣', '8': '8️⃣', '9': '9️⃣'
}


def create_prompt_response(min_length=5, max_length=10):
    length = random.randint(min_length, max_length)
    chars = random.choices(list(char_to_emoji_map.keys()), k=length)
    dash_string = "-".join(chars)
    prompt = prompt_template.format(dash_string)

    steps = []
    final = ""
    for c in chars:
        emoji = char_to_emoji_map[c]
        steps.append(f"{c} → {emoji}")
        final += emoji

    steps.append(f"Final emoji string: \\boxed{{{final}}}")
    response = "\n".join(steps)

    return prompt, response, final


def make_dataset(path="~/data/speek_emoji", total=10000):
    path = os.path.expanduser(path)
    print(f"数据集保存路径为:", path)
    os.makedirs(path, exist_ok=True)
    # 创建子目录
    for subdir in ["sft", "rl"]:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)

    sft = {"prompt": [], "response": []}
    rl = {"prompt": [], "data_source": [], "ability": [], "reward_model": [], "extra_info": []}

    for _ in range(total):
        prompt, response, ground_truth = create_prompt_response()
        sft["prompt"].append(prompt)
        sft["response"].append(response)
        rl["prompt"].append([{"role": "user", "content": prompt}])
        rl["data_source"].append("char_to_emoji")
        rl["ability"].append("mapping")
        rl["reward_model"].append({"style": "rule", "ground_truth": ground_truth})
        rl["extra_info"].append({"response": response})

    # split
    split = int(0.9 * total)
    pd.DataFrame(sft).iloc[:split].to_parquet(os.path.join(path, "sft/train.parquet"))
    pd.DataFrame(sft).iloc[split:].to_parquet(os.path.join(path, "sft/test.parquet"))
    pd.DataFrame(rl).iloc[:split].to_parquet(os.path.join(path, "rl/train.parquet"))
    pd.DataFrame(rl).iloc[split:].to_parquet(os.path.join(path, "rl/test.parquet"))

    print("\n📌 示例 SFT 样本:")
    print(json.dumps(pd.DataFrame(sft).head(2).to_dict(orient="records"), indent=2, ensure_ascii=False))

    print("\n📌 示例 RL 样本:")
    print(json.dumps(pd.DataFrame(rl).head(2).to_dict(orient="records"), indent=2, ensure_ascii=False))

    print("✅ Dataset generated at:", path)


if __name__ == "__main__":
    make_dataset()
