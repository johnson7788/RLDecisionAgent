#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/7/21 17:06
# @File  : reward_function.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 强化学习中的奖励函数， 判断是否回答正确，回答正确，那么奖励+1，否则为0


from verl.utils.reward_score import math

def char_to_emoji_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    try:
        last_boxed = math.last_boxed_only_string(solution_str)
        if last_boxed is None:
            return 0
        predicted = math.remove_boxed(last_boxed)
        return 1 if predicted == ground_truth else 0
    except Exception:
        print("Error in reward function:", ground_truth, solution_str)
        return 0
