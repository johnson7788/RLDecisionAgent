import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List

import openai
from datasets import Dataset
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from openpipe import AsyncOpenPipe
from transformers.models.auto.tokenization_auto import AutoTokenizer
from utils import cache, prompt_for_title, pull_data, score_title

import art
from art.local import LocalBackend
from art.utils import iterate_dataset, limit_concurrency

# 加载 .env 文件中的环境变量（如 API Key）
load_dotenv()

# --- 全局配置参数 ---
MODEL_NAME = "001"                      # 本地训练模型的名称（实验 ID）
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct" # 基础预训练模型
MAX_COMPLETION_LENGTH = 100             # 模型生成标题的最大长度
MAX_PROMPT_LENGTH = 8192 - MAX_COMPLETION_LENGTH  # 最大输入 token 长度（留出输出空间）
LEARNING_RATE = 1.2e-5                  # 学习率
GROUPS_PER_STEP = 1                     # 每一步采样的 group 数
EVAL_STEPS = 50                         # 每隔多少步做一次验证
VAL_SET_SIZE = 100                      # 验证集大小
TRAINING_DATASET_SIZE = 5000            # 训练集大小
PROJECT = "hn_title_generation"         # 项目名称
NUM_EPOCHS = 1                          # 训练 epoch 数
NUM_GENERATIONS = 6                     # 每个样本生成多少个标题（并行采样数量）


# --- 数据预处理 ---
def filter_on_length(data: Dataset, max_length: int, tokenizer_name: str) -> Dataset:
    """
    根据 prompt 长度过滤数据，确保不会超过模型最大输入限制。
    """
    print(f"Filtering dataset for max prompt length: {max_length} using tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def check_length(x):
        # 确保 prompt 是一个消息列表（system/user 格式）
        if not isinstance(x.get("prompt"), list):
            print(f"Warning: Skipping row with invalid prompt format: {x}")
            return False
        try:
            # 计算 token 数量
            tokenized_len = len(
                tokenizer.apply_chat_template(
                    x["prompt"], tokenize=True, add_generation_prompt=True
                )
            )
            return tokenized_len <= max_length
        except Exception as e:
            # 如果出错，跳过该样本
            print(f"Warning: Error tokenizing prompt, skipping row. Error: {e}, Prompt: {x['prompt']}")
            return False

    len_before = len(data)
    data = data.filter(check_length)
    len_after = len(data)
    print(f"Samples before length filtering: {len_before}, samples after: {len_after}")

    # 打印提示：是否过滤过多
    if len_after == 0 and len_before > 0:
        print("Warning: All samples were filtered out. Check MAX_PROMPT_LENGTH and tokenizer.")
    elif len_after < len_before * 0.5:
        print(f"Warning: More than 50% of samples filtered out ({len_before - len_after} samples).")
    return data


@cache.cache()
async def load_data(
    split: str = "train",
    max_items: int = 10,
    min_score: int = 20,
    max_length: int = 8192,
    tokenizer_name: str = BASE_MODEL,
) -> Dataset:
    """
    加载并预处理数据集：
    1. 拉取数据
    2. 过滤掉没有正文的样本
    3. 将正文转换为 prompt（system + user 消息）
    4. 按长度过滤
    """
    print(f"Loading data for split: {split}, max_items: {max_items}, tokenizer: {tokenizer_name}")
    data = pull_data(split=split, max_items=max_items, min_score=min_score)
    if not data:
        raise ValueError(f"No data loaded for split {split}. Check pull_data function.")

    # 过滤掉没有有效 scraped_body 的样本
    def check_scraped_body(x):
        body = x.get("scraped_body")
        return isinstance(body, str) and len(body.strip()) > 0

    data = data.filter(check_scraped_body)
    if not data:
        raise ValueError(f"No data remaining after filtering for valid 'scraped_body' in split {split}.")

    # 转换为 prompt 格式
    data = data.map(
        lambda x: {
            "prompt": prompt_for_title(x["scraped_body"]),  # 生成 system+user 消息
            "row": x,  # 保存原始数据行
        }
    )
    return filter_on_length(data, max_length, tokenizer_name)


# --- 打分函数 ---
@limit_concurrency(10)  # 限制并发请求数量
async def call_score_title(row_with_title: Dict[str, Any]) -> float:
    """调用 RM 模型对标题打分"""
    return await score_title(row_with_title, "rm")


async def check_title_matches_body(client: openai.AsyncOpenAI, body: str, title: str) -> int:
    """
    使用 LLM 验证生成的标题是否与正文内容匹配（不能有无根据的夸大/虚假）。
    返回 1 表示匹配，0 表示不匹配。
    """
    system_prompt = "You are a moderator for Hacker News. You are given the body of an article, as well as a proposed title. You are to determine whether the title makes any claims that are not substantiated by the article body. If there are any unsubstantiated claims, you should return False. Otherwise, you should return True. Only return False or True, no other text."
    user_prompt = f"<article>{body}</article>\n<proposed_title>{title}</proposed_title>"

    messages: Iterable[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = await client.chat.completions.create(
            model=BASE_MODEL,  # 使用基础模型进行验证
            messages=messages,
            max_tokens=5,  # 只需要输出 True / False
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content:
            content_cleaned = content.strip().lower()
            if content_cleaned.startswith("true"):
                return 1
            elif content_cleaned.startswith("false"):
                return 0
            else:
                print(f"Warning: Unexpected validation response: '{content}'. Defaulting to mismatch (0).")
                return 0
        else:
            print("Warning: Empty validation response. Defaulting to mismatch (0).")
            return 0
    except Exception as e:
        print(f"Error during title validation API call: {e}. Defaulting to mismatch (0).")
        return 0


# --- rollout（一次训练采样流程） ---
async def rollout(
    client: openai.AsyncOpenAI,
    op_client: AsyncOpenPipe,
    model_name: str,
    prompt: Iterable[ChatCompletionMessageParam],
    row: Dict[str, Any],
    global_step: int,
    epoch: int,
) -> art.Trajectory:
    """
    1. 用模型生成标题
    2. 验证标题是否与正文一致
    3. 用 RM 打分
    4. 计算奖励并返回 rollout 轨迹
    """
    metrics = {}
    requested_at = datetime.now()

    # Step 1: 生成标题
    chat_completion = await client.chat.completions.create(
        messages=prompt,
        model=model_name,
        max_tokens=MAX_COMPLETION_LENGTH,
        temperature=1,
        logprobs=True,
    )
    received_at = datetime.now()
    choice = chat_completion.choices[0]
    generated_title = choice.message.content
    if not generated_title:
        print("Warning: Empty title generated.")
        generated_title = ""  # Assign empty string if None or empty

    metrics["length"] = len(generated_title)

    # Step 2: 验证标题与正文是否匹配
    title_matches = await check_title_matches_body(client, row["scraped_body"], generated_title)
    metrics["matches"] = title_matches

    # Step 3: 打分
    row_with_title = {**row, "title": generated_title}
    rm_score = await asyncio.wait_for(call_score_title(row_with_title), timeout=30)
    metrics["rm"] = rm_score

    # Step 4: 计算奖励
    final_reward = 0.0 if title_matches == 0 else rm_score

    # 记录 rollout 过程
    messages_and_choices = [*prompt, choice]

    await op_client.report(
        requested_at=requested_at.timestamp(),
        received_at=received_at.timestamp(),
        req_payload={
            "model": model_name,
            "messages": prompt,
            "metadata": {
                "type": "art_rollout",
                "split": row["split"],
                "step": global_step,
                "epoch": epoch,
                "dataset_id": row["id"],
                **metrics,
            },
        },
        resp_payload=chat_completion,
        status_code=200,
    )

    trajectory = art.Trajectory(
        messages_and_choices=messages_and_choices,
        reward=final_reward,
        metrics=metrics,
    )
    return trajectory


# --- 主训练循环 ---
async def main():
    # 初始化 ART 本地后端与模型
    backend = LocalBackend()
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(gpu_memory_utilization=0.75),
            peft_args=art.dev.PeftArgs(lora_alpha=8),
            trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
        ),
    )
    await model.register(backend)
    op_client = AsyncOpenPipe(api_key=os.getenv("OPENPIPE_API_KEY"))

    # 加载训练集与验证集
    print("Loading training data...")
    train_dataset = await load_data("train", TRAINING_DATASET_SIZE, MAX_PROMPT_LENGTH, BASE_MODEL)
    print("Loading validation data...")
    val_dataset = await load_data("val", VAL_SET_SIZE, MAX_PROMPT_LENGTH, BASE_MODEL)

    if not train_dataset or not val_dataset:
        raise ValueError("Failed to load datasets. Exiting.")

    # 转为列表方便遍历
    val_data_list: List[Dict[str, Any]] = list(val_dataset)
    train_data_list: List[Dict[str, Any]] = list(train_dataset)

    print(f"Training data size: {len(train_data_list)}")
    print(f"Validation data size: {len(val_data_list)}")

    # 获取可调用的 OpenAI 客户端
    openai_client = model.openai_client()

    # 开始训练循环
    start_step = await model.get_step()
    print(f"Starting training from global step {start_step}")

    data_iterator = iterate_dataset(
        dataset=train_data_list,
        groups_per_step=GROUPS_PER_STEP,
        num_epochs=NUM_EPOCHS,
        initial_step=start_step,
        use_tqdm=True,
    )

    for batch in data_iterator:
        # rollout 采样多个标题
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(openai_client, op_client, MODEL_NAME, bi["prompt"], bi["row"], batch.step, batch.epoch)
                    for _ in range(NUM_GENERATIONS)
                )
                for bi in batch.items
            )
        )

        # 过滤掉无效轨迹
        valid_train_groups = []
        for group in train_groups:
            valid_group = [traj for traj in group if isinstance(traj, art.Trajectory)]
            if len(valid_group) > 1:
                valid_train_groups.append(valid_group)

        if not valid_train_groups:
            print(f"Warning: No valid trajectories generated for step {batch.step}. Skipping tune step.")
            continue

        # 更新模型参数
        await model.train(valid_train_groups, config=art.TrainConfig(learning_rate=LEARNING_RATE))

        # 每隔 EVAL_STEPS 步进行验证
        if batch.step > 0 and batch.step % EVAL_STEPS == 0:
            print(f"\n--- Evaluating at Step {batch.step} ---")
            val_trajectories = await art.gather_trajectories(
                (
                    rollout(openai_client, op_client, MODEL_NAME, item["prompt"], item["row"], batch.step, batch.epoch)
                    for item in val_data_list
                ),
                pbar_desc="val",
            )
            await model.log(val_trajectories)
            await model.delete_checkpoints()

    print("Training finished.")


if __name__ == "__main__":
    asyncio.run(main())  # 启动主训练任务
