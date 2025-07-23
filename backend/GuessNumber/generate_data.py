#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/7/19 02:57
# @File  : prepare_data.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 成多轮的对话的数据集， 猜数字， 和用户一起玩猜数字，假如你现在心里想的数字是58，用户说任意一个数字，你只会说大了，小了，或者🎉Bingo正确，
# 超过一定轮次，回答： 没猜中哦，正确答案是 {target}


import random
import json


def generate_guess_game_data(n=100, num_range=(1, 100), max_turns=6):
    data = []
    for _ in range(n):
        target = random.randint(*num_range)
        turns = []
        # 添加 system 提示
        turns.append({
            "role": "system",
            "content": f"你能和用户一起玩猜数字，假如你现在心里想的数字是{target}，用户说任意一个数字，你只会说大了，小了，或者🎉Bingo正确"
        })
        low, high = num_range
        guessed_correctly = False

        for attempt in range(max_turns):
            guess = random.randint(low, high)
            turns.append({"role": "user", "content": str(guess)})
            if guess < target:
                turns.append({"role": "assistant", "content": "小了"})
                low = guess + 1
            elif guess > target:
                turns.append({"role": "assistant", "content": "大了"})
                high = guess - 1
            else:
                turns.append({"role": "assistant", "content": "🎉Bingo正确"})
                guessed_correctly = True
                break

        # 超过最大尝试次数还没猜中
        if not guessed_correctly:
            turns.append({
                "role": "assistant",
                "content": f"没猜中哦，正确答案是 {target}"
            })

        data.append({"messages": turns})

    return data


# 写入为 JSON 文件
if __name__ == "__main__":
    number =1000
    output_data = generate_guess_game_data(n=number)
    print(json.dumps(output_data, ensure_ascii=False, indent=2))
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已生成 guess_number_dataset.json，共 {number} 组对话。")
