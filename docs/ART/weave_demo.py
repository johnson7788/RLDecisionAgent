#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/23 07:10
# @File  : weave_demo.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

# pip install -U wandb weave
import os, time
import wandb
import weave

# 若使用自建 W&B：在 init 之前设置
# os.environ["WANDB_BASE_URL"] = "http://192.168.100.8:3005"
# export WANDB_API_KEY=...  # 或提前用 `wandb login --host=...`

run = wandb.init(project="weave-demo", config={"lr": 1e-3})

weave.init("weave-demo")  # 也可用 "entity/project"；将使用同一账号并与当前 run 关联

@weave.op()   # 这个函数的每次调用都会被 Weave 追踪
def score(text: str) -> float:
    time.sleep(0.1)
    return len(text) / 10.0

for step, s in enumerate(["hello", "world", "weave + wandb"]):
    val = score(s)                     # 触发 Weave trace
    wandb.log({"metric/score": val}, step=step)

run.finish()
