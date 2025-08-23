#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/23
# @File  : model_test.py
# @Author: Gemini
# @Desc  : 对 'hn_title_generation' 项目训练好的模型进行测试。

import asyncio
from dotenv import load_dotenv

import art
from art.local import LocalBackend
from train import load_data
# 从训练脚本中导入核心配置，确保测试与训练环境一致
from train import MODEL_NAME, PROJECT, BASE_MODEL, MAX_PROMPT_LENGTH, MAX_COMPLETION_LENGTH

# 加载环境变量
load_dotenv()

# --- 测试配置 ---
# 使用验证集（'val'）进行测试
TEST_SPLIT = "val"
# 加载多少条数据进行测试
MAX_TEST_ITEMS = 20


async def main():
    """
    主测试函数：
    1. 初始化并注册在 `train.py` 中训练的模型。
    2. 加载测试数据集。
    3. 使用模型对每条数据生成标题。
    4. 打印原文、生成标题和原始标题以供对比。
    """
    print(f"开始测试模型: {MODEL_NAME} (项目: {PROJECT})")

    # 1. 初始化 ART 本地后端与可训练模型
    # 这里的 name, project, base_model 必须与训练时完全一致
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
        # 关键：为保证加载正确，测试时需提供与训练时相同的内部配置
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(gpu_memory_utilization=0.75),
            peft_args=art.dev.PeftArgs(lora_alpha=8),
            trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
        ),
    )
    backend = LocalBackend(in_process=True)
    await model.register(backend)

    # 2. 加载测试数据
    print(f"正在加载 {TEST_SPLIT} 分割的测试数据（最多 {MAX_TEST_ITEMS} 条）...")
    try:
        test_dataset = await load_data(
            split=TEST_SPLIT,
            max_items=MAX_TEST_ITEMS,
            max_length=MAX_PROMPT_LENGTH,
            tokenizer_name=BASE_MODEL,
        )
    except ValueError as e:
        print(f"错误：加载数据失败: {e}")
        print("请确保数据源可用，并且 'utils.py' 中的 'pull_data' 函数配置正确。")
        return

    if not test_dataset or len(test_dataset) == 0:
        print("错误：未能加载任何测试数据。程序退出。")
        return

    test_data_list = list(test_dataset)
    print(f"成功加载 {len(test_data_list)} 条测试数据。")

    # 从注册好的模型获取 OpenAI 兼容的客户端
    client = model.openai_client()

    # 3. 遍历测试集并生成结果
    for i, scenario in enumerate(test_data_list):
        print(f"\n--- 测试样本 {i + 1}/{len(test_data_list)} ---")

        original_body = scenario.get("row", {}).get("scraped_body", "[无内容]")
        original_title = scenario.get("row", {}).get("title", "[无标题]")
        prompt = scenario.get("prompt")

        # 打印原始信息
        print(f"原始文章 (摘要): {original_body[:400].strip()}...")
        print(f"原始参考标题: {original_title}")

        if not prompt:
            print("警告：该样本缺少 'prompt'，跳过生成。")
            continue

        # 4. 调用模型生成标题
        print("正在生成标题...")
        try:
            chat_completion = await client.chat.completions.create(
                messages=prompt,
                model=MODEL_NAME,  # 关键：使用我们微调过的模型
                max_tokens=MAX_COMPLETION_LENGTH,
                temperature=0.7,  # 使用适中的 temperature 以获得有创造力但不过于随机的输出
                stop=None,  # 可以根据需要设置停止词
            )
            generated_title = chat_completion.choices[0].message.content.strip()
            print(f"✅ 模型生成标题: {generated_title}")

        except Exception as e:
            print(f"❌ 生成标题时出错: {e}")

    print("\n--- 测试完成 ---")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())