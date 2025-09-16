#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/16 11:28
# @File  : glaive_tool_process.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :


import json, sys
inp, out = "glaive_toolcall.jsonl", "glaive_toolcall.fixed.jsonl"
with open(inp, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        tools = obj.get("tools", None)
        # 统一把字符串/对象归一成 list，其他非法值置为 None
        if isinstance(tools, str):
            try:
                parsed = json.loads(tools)
            except Exception:
                parsed = None
            if isinstance(parsed, dict): parsed = [parsed]
            if not isinstance(parsed, list): parsed = None
            obj["tools"] = parsed
        elif isinstance(tools, dict):
            obj["tools"] = [tools]
        elif not isinstance(tools, list):
            obj["tools"] = None
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
print("Wrote", out)