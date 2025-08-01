#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/1 22:49
# @File  : check_gpu_avaiables.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import torch

def check_gpus():
    gpu_count = torch.cuda.device_count()
    print(f"可用GPU数量: {gpu_count}")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 单位GB
        print(f"GPU {i}: {gpu_name}, 计算能力: {capability}, 显存: {total_mem:.2f} GB")

    if gpu_count == 0:
        print("当前系统没有检测到可用的GPU。")
    else:
        print("GPU检查完成。")

if __name__ == "__main__":
    check_gpus()
