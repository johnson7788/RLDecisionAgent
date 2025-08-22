#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/22 20:12
# @File  : prompt.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

rollout_system_prompt = """
You are an MCP (Model Context Protocol) agent.
You have access to MCP tools through the server. Use them to complete your task.
You have a total of {max_turns} turns. Only use tool calls.
Call 'complete_task' when finished.
"""