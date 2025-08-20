import asyncio
import json
import os
import random
import re
from typing import TypedDict

from dotenv import load_dotenv

import art
from art.local import LocalBackend

# 加载环境变量（从 .env 文件中读取）
load_dotenv()


# 定义一个类型字典，用于存储谜题的结构
class TemporalCluePuzzle(TypedDict):
    num_clues: int  # 线索数量
    prompt: str  # 提示文本
    solution: dict[str, str]  # 谜题的答案（键值对）


# 从本地 JSON 文件中加载谜题数据
puzzles_path = os.path.join(
    os.path.dirname(__file__),  "puzzles.json"
)

# 读取并解析谜题
puzzles: list[TemporalCluePuzzle] = json.loads(open(puzzles_path).read())
val_puzzles = puzzles[:64]  # 验证集
test_puzzles = puzzles[64:128]  # 测试集
train_puzzles = puzzles[128:]  # 训练集
random.seed(42)  # 固定随机种子，保证结果可复现
random.shuffle(train_puzzles)  # 打乱训练集


# 定义异步 rollout 函数：使用模型生成回答并计算奖励
async def rollout(model: art.Model, puzzle: TemporalCluePuzzle) -> art.Trajectory:
    # 构造输入消息
    messages: art.Messages = [{"role": "user", "content": puzzle["prompt"]}]

    # 调用 OpenAI API 进行推理
    client = model.openai_client()
    chat_completion = await client.chat.completions.create(
        messages=messages, model=model.name
    )

    # 获取模型生成的回答
    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)

    # 检查回答中是否包含正确答案
    num_correct = 0
    for key, value in puzzle["solution"].items():
        if matches := re.findall(rf"{key}\. ([A-Za-z \.:-]+)", content):
            match = matches[-1]
            if match.strip().lower() == value.lower():
                num_correct += 1

    # 计算奖励（准确率）
    reward = acc = num_correct / len(puzzle["solution"])

    # 返回一个 Trajectory（轨迹，包含消息、奖励和评估指标）
    return art.Trajectory(
        messages_and_choices=[*messages, choice], reward=reward, metrics={"acc": acc}
    )


# 主函数：模型训练逻辑
async def main():
    # 定义可训练模型
    model = art.TrainableModel(
        name="001",  # 模型名称
        project="temporal-clue",  # 项目名称
        base_model="Qwen/Qwen2.5-7B-Instruct",  # 基础模型
        _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},  # GPU 配置
    )

    backend = LocalBackend()  # 使用本地后端
    # await backend._experimental_pull_from_s3(model)  # 从 S3 下载模型
    await model.register(backend)  # 注册模型到后端

    stride = 4  # 每次训练使用的步长
    for i in range(await model.get_step(), 1_000):  # 从当前步数训练到 1000 步
        # 并发收集验证集和训练集的轨迹
        val_groups, train_groups = await asyncio.gather(
            art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(rollout(model, puzzle) for _ in range(2))
                    for puzzle in val_puzzles
                ),
                pbar_desc="val",  # 验证集进度条
            ),
            art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(rollout(model, puzzle) for _ in range(50))
                    for puzzle in train_puzzles[i * stride: (i + 1) * stride]
                ),
                pbar_desc="train",  # 训练集进度条
            ),
        )

        # 记录验证结果
        await model.log(val_groups)
        # 删除旧的检查点，节省存储
        await model.delete_checkpoints()
        # 将模型推送到 S3 保存
        # await backend._experimental_push_to_s3(model)
        # 使用收集到的训练数据进行训练
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=5e-5),  # 设置学习率
        )


# 入口函数：运行主程序
if __name__ == "__main__":
    asyncio.run(main())
