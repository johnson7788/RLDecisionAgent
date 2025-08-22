#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/22 17:05
# @File  : data_convert.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import json
import argparse
import math
import random
from pathlib import Path
from typing import List, Dict

def load_and_convert(input_path: Path, difficulty: int) -> List[Dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")
    data = []
    total, skipped = 0, 0
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                skipped += 1
                continue
            data.append({"task": question.strip(), "difficulty": difficulty})
    print(f"读取完成：{total} 行，有效 {len(data)} 行，跳过 {skipped} 行。")
    return data

def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {len(rows)} 行 → {path.resolve()}")

def main():
    parser = argparse.ArgumentParser(
        description="将 questions.jsonl 转为 {task, difficulty}，并写入 scenarios/train/val 三个JSONL"
    )
    parser.add_argument("-i", "--input", default="questions.jsonl", help="输入 JSONL（默认：questions.jsonl）")
    parser.add_argument("--scenarios", default="scenarios.jsonl", help="输出场景样例文件（默认：scenarios.jsonl）")
    parser.add_argument("--train", default="train.jsonl", help="输出训练集文件（默认：train.jsonl）")
    parser.add_argument("--val", default="val.jsonl", help="输出验证集文件（默认：val.jsonl）")
    parser.add_argument("-d", "--difficulty", type=int, default=1, help="难度（默认：1）")
    parser.add_argument("--scenarios-count", type=int, default=10, help="scenarios.jsonl 写入条数上限（默认：10）")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例（默认：0.9）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认：42）")
    args = parser.parse_args()

    input_path = Path(args.input)
    scenarios_path = Path(args.scenarios)
    train_path = Path(args.train)
    val_path = Path(args.val)

    # 1) 读取并转换
    data = load_and_convert(input_path, args.difficulty)
    n = len(data)

    # 2) 写 scenarios.jsonl（前 n 或最多 args.scenarios_count 条）
    k = min(n, args.scenarios_count)
    write_jsonl(scenarios_path, data[:k])

    # 3) 90/10 划分 train/val（对全部数据划分；如需排除 scenarios 的 10 条，可另行调整）
    rnd = random.Random(args.seed)
    indices = list(range(n))
    rnd.shuffle(indices)

    train_size = int(math.floor(n * args.train_ratio))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_rows = [data[i] for i in train_idx]
    val_rows = [data[i] for i in val_idx]

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    print(f"划分比例：train {len(train_rows)}/{n}，val {len(val_rows)}/{n}（train_ratio={args.train_ratio}）")
    if n > 0 and len(val_rows) == 0:
        print("注意：样本量较小，val 集为空。如需至少 1 条验证数据，可把 --train-ratio 调小一点。")

if __name__ == "__main__":
    main()
