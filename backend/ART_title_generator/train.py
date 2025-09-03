"""
HN 标题生成（Hacker News Title Generation）- 使用 ART 做强化学习微调
-----------------------------------------------------------------------
本脚本演示了一个端到端的训练流程：
1) 加载与预处理数据（将文章正文转换为 Chat 形式的 prompt）
2) 用可训练模型生成候选标题（rollout）
3) 使用校验器（基础模型）判定标题是否与正文一致（防止夸大/幻觉）
4) 使用 RM（Reward Model）对标题打分
5) 组合得到奖励并更新策略（可训练模型）
6) 周期性评估与日志上报（基于 ART 的日志与 checkpoint）

依赖与环境：
- 需要在项目根目录放置 .env 文件（如有 OPENAI_API_KEY / HF_TOKEN 等，可在此声明）。
- 依赖的内部工具/库：art（训练框架）、utils（数据与缓存工具）。

注意：
- MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH 不应超过基础模型的最大上下文长度。
- 本脚本默认以 Qwen2.5-7B-Instruct 作为 BASE_MODEL。
"""
import os
import asyncio
from datetime import datetime  # 可保留以备进一步自定义日志
from typing import Any, Dict, Iterable, List
import wandb
import openai
from datasets import Dataset
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from transformers.models.auto.tokenization_auto import AutoTokenizer
from utils import cache, prompt_for_title, pull_data, score_title

import art
from art.local import LocalBackend
from art.utils import iterate_dataset, limit_concurrency

# 读取 .env 文件中的环境变量（例如 OPENAI_API_KEY / HF_TOKEN 等）
load_dotenv()

# --- 全局配置参数（可根据需要调整） ---
MODEL_NAME = os.environ["MODEL_NAME"]                   # 可训练模型的标识（本地实验/版本号）
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct" # 用于推断/校验的基础指令模型
MAX_COMPLETION_LENGTH = 100             # 生成标题允许的最大 token 数（输出上限）
# 预留输出 token 空间后，限制输入 token 的最大长度（防止总长度超上下文）
MAX_PROMPT_LENGTH = 8192 - MAX_COMPLETION_LENGTH
LEARNING_RATE = 1.2e-5                  # 学习率（策略更新）
GROUPS_PER_STEP = 1                      # 每个训练 step 采样的 group 数量
EVAL_STEPS = 50                          # 每隔多少个 step 做一次验证评估
VAL_SET_SIZE = 100                       # 验证集大小
TRAINING_DATASET_SIZE = 5000             # 训练集大小
PROJECT = os.environ["PROJECT"]            # 项目标识（ART 内部使用）
NUM_EPOCHS = 1                           # 训练轮数（遍历数据的次数）
NUM_GENERATIONS = 6                      # 单样本并行生成多少个标题（用于探索）


# --- 数据预处理 ---
def filter_on_length(data: Dataset, max_length: int, tokenizer_name: str) -> Dataset:
    """
    根据 prompt 的 token 长度过滤数据，以避免超过模型最大上下文长度。
    - 使用给定 tokenizer（与 BASE_MODEL 对齐）将聊天消息模板化并 tokenization。
    - 若某条数据无法成功 tokenization，或长度超限，则丢弃。
    """
    print(f"按最大 Prompt 长度过滤数据：{max_length}（分词器：{tokenizer_name}）")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def check_length(x):
        # 期望 prompt 为消息列表（[{role, content}, ...]），否则跳过
        if not isinstance(x.get("prompt"), list):
            print(f"警告：跳过一条无效的 prompt 格式数据：{x}")
            return False
        try:
            # 使用 chat 模板计算 token 数，add_generation_prompt=True 表示补齐 assistant 起始标记
            tokenized_len = len(
                tokenizer.apply_chat_template(
                    x["prompt"], tokenize=True, add_generation_prompt=True
                )
            )
            return tokenized_len <= max_length
        except Exception as e:
            # 遇到异常（例如非法字符/格式），保守丢弃该样本
            print(f"警告：对 prompt 分词失败，已跳过。错误：{e}，Prompt：{x['prompt']}")
            return False

    len_before = len(data)
    data = data.filter(check_length)
    len_after = len(data)
    print(f"长度过滤前样本数：{len_before}，过滤后：{len_after}")

    # 提示过滤比例，防止误设阈值导致样本被过度丢弃
    if len_after == 0 and len_before > 0:
        print("警告：所有样本均被过滤。请检查 MAX_PROMPT_LENGTH 与分词器设置。")
    elif len_after < len_before * 0.5:
        print(f"警告：超过 50% 的样本被过滤（{len_before - len_after} 条）。")
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
    1) 拉取数据（pull_data）
    2) 过滤掉没有正文 scraped_body 的样本
    3) 将正文转换为 chat prompt（system + user）
    4) 基于长度做二次过滤

    说明：
    - 使用 @cache.cache() 装饰器缓存函数结果，避免重复拉取与处理，同参数下二次调用会复用。
    - min_score 可用于下游数据选择（由 pull_data 内部决定如何应用）。
    """
    print(f"加载数据：split={split}，max_items={max_items}，tokenizer={tokenizer_name}")
    data = pull_data(split=split, max_items=max_items, min_score=min_score)
    if not data:
        raise ValueError(f"No data loaded for split {split}. Check pull_data function.")

    # 过滤掉没有有效 scraped_body 的样本（空/None/仅空白）
    def check_scraped_body(x):
        body = x.get("scraped_body")
        return isinstance(body, str) and len(body.strip()) > 0

    data = data.filter(check_scraped_body)
    if not data:
        raise ValueError(f"No data remaining after filtering for valid 'scraped_body' in split {split}.")

    # 将正文转换为 prompt：prompt_for_title 会构造 system+user 的消息列表
    data = data.map(
        lambda x: {
            "prompt": prompt_for_title(x["scraped_body"]),  # 用于生成标题的聊天消息
            "row": x,  # 保留原始数据，供后续评分与日志字段使用
        }
    )
    return filter_on_length(data, max_length, tokenizer_name)


# --- 打分函数（Reward Model） ---
@limit_concurrency(10)  # 限制并发，避免打爆打分服务或网络资源
async def call_score_title(row_with_title: Dict[str, Any]) -> float:
    """调用 RM 模型对标题打分，返回一个浮点分数（越高越好）。"""
    return await score_title(row_with_title, "rm")


async def check_title_matches_body(client: openai.AsyncOpenAI, body: str, title: str) -> int:
    """
    使用基础 LLM 验证生成的标题是否与正文内容匹配：
    - 若标题存在未被正文支持的断言/夸大，则视为不匹配，返回 0
    - 否则返回 1
    该校验作为安全阀，避免奖励推动模型学到“博眼球”的幻觉标题。
    """
    system_prompt = "You are a moderator for Hacker News. You are given the body of an article, as well as a proposed title. You are to determine whether the title makes any claims that are not substantiated by the article body. If there are any unsubstantiated claims, you should return False. Otherwise, you should return True. Only return False or True, no other text."
    user_prompt = f"<article>{body}</article>\n<proposed_title>{title}</proposed_title>"

    messages: Iterable[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        # 使用基础模型进行“判真伪”校验：输出期望仅为 True/False
        response = await client.chat.completions.create(
            model=BASE_MODEL,
            messages=messages,
            max_tokens=5,  # 极短输出即可
            temperature=0.0,  # 关闭采样，保证可重复性
        )
        content = response.choices[0].message.content
        if content:
            content_cleaned = content.strip().lower()
            if content_cleaned.startswith("true"):
                return 1
            elif content_cleaned.startswith("false"):
                return 0
            else:
                # 未按预期返回 True/False，保守视为不匹配
                print(f"警告：校验返回了非预期结果：'{content}'。默认判为不匹配（0）。")
                return 0
        else:
            print("警告：校验返回为空。默认判为不匹配（0）。")
            return 0
    except Exception as e:
        # 校验失败时不阻断训练流程，返回不匹配，避免奖励错误扩大
        print(f"错误：标题一致性校验 API 调用失败：{e}。默认判为不匹配（0）。")
        return 0


# --- rollout（单步采样与打分） ---
async def rollout(
    client: openai.AsyncOpenAI,
    model_name: str,
    prompt: Iterable[ChatCompletionMessageParam],
    row: Dict[str, Any],
    global_step: int,
    epoch: int,
) -> art.Trajectory:
    """
    执行一次完整的采样-评估流程：
    1) 用可训练模型生成候选标题（带 logprobs 便于训练）
    2) 用基础模型校验标题与正文一致性
    3) 用 RM 打分（质量评估）
    4) 将一致性与 RM 分结合并得到最终奖励，返回 Trajectory
    """
    metrics = {}
    requested_at = datetime.now()

    # Step 1: 生成标题（带 logprobs 便于后续策略梯度/对比学习等）
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
        # 极端情况下返回空，避免后续抛异常
        print("警告：生成的标题为空。")
        generated_title = ""

    metrics["length"] = len(generated_title)  # 记录长度以便分析

    # Step 2: 一致性校验（1=匹配，0=不匹配）
    title_matches = await check_title_matches_body(client, row["scraped_body"], generated_title)
    metrics["matches"] = title_matches

    # Step 3: 打分（RM）
    row_with_title = {**row, "title": generated_title}
    rm_score = await asyncio.wait_for(call_score_title(row_with_title), timeout=30)
    metrics["rm"] = rm_score

    # Step 4: 组合奖励策略：若不匹配则零奖励，否则使用 RM 分数
    final_reward = 0.0 if title_matches == 0 else rm_score

    # === 新增：把关键指标也放进 metrics，便于后续统一聚合/打点 ===
    metrics["reward"] = float(final_reward)
    metrics["step"] = int(global_step)
    metrics["epoch"] = int(epoch)

    # 打包为 ART 轨迹对象（仍保留 messages 与 choice 供训练/审计）
    messages_and_choices = [*prompt, choice]
    trajectory = art.Trajectory(
        messages_and_choices=messages_and_choices,
        reward=final_reward,
        metrics=metrics,
    )
    return trajectory


# --- 主训练循环 ---
async def main():
    # === wandb 初始化（在做任何训练前调用）===
    run = wandb.init(
        project=PROJECT,
        name=f"{MODEL_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=dict(
            MODEL_NAME=MODEL_NAME,
            BASE_MODEL=BASE_MODEL,
            MAX_COMPLETION_LENGTH=MAX_COMPLETION_LENGTH,
            MAX_PROMPT_LENGTH=MAX_PROMPT_LENGTH,
            LEARNING_RATE=LEARNING_RATE,
            GROUPS_PER_STEP=GROUPS_PER_STEP,
            EVAL_STEPS=EVAL_STEPS,
            VAL_SET_SIZE=VAL_SET_SIZE,
            TRAINING_DATASET_SIZE=TRAINING_DATASET_SIZE,
            NUM_EPOCHS=NUM_EPOCHS,
            NUM_GENERATIONS=NUM_GENERATIONS,
        ),
    )
    # 统一 step 轴，让图表更干净
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")

    # 1) 初始化 ART 本地后端与可训练模型
    backend = LocalBackend()
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(gpu_memory_utilization=0.75),   # 显存占用上限
            peft_args=art.dev.PeftArgs(lora_alpha=8),                  # LoRA 等 PEFT 相关参数
            trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),       # 梯度裁剪，稳定训练
        ),
    )
    await model.register(backend)

    # 2) 加载训练/验证数据集（含过滤与 prompt 构造）
    print("正在加载训练集...")
    train_dataset = await load_data(
        split="train",
        max_items=TRAINING_DATASET_SIZE,
        max_length=MAX_PROMPT_LENGTH,
        tokenizer_name=BASE_MODEL,
    )
    print("正在加载验证集...")
    val_dataset = await load_data(
        "val",
        max_items=VAL_SET_SIZE,
        max_length=MAX_PROMPT_LENGTH,
        tokenizer_name=BASE_MODEL,
    )

    # 转为列表方便迭代/分组
    val_data_list: List[Dict[str, Any]] = list(val_dataset)
    train_data_list: List[Dict[str, Any]] = list(train_dataset)

    print(f"训练集样本数：{len(train_data_list)}")
    print(f"验证集样本数：{len(val_data_list)}")

    # 从 TrainableModel 获取与之绑定的 OpenAI 兼容客户端
    openai_client = model.openai_client()

    # 3) 训练循环入口：从上次中断的 step 继续
    start_step = await model.get_step()
    print(f"从全局 step {start_step} 开始训练")

    # iterate_dataset 会按 epoch/group 切分数据，并提供进度条
    data_iterator = iterate_dataset(
        dataset=train_data_list,
        groups_per_step=GROUPS_PER_STEP,
        num_epochs=NUM_EPOCHS,
        initial_step=start_step,
        use_tqdm=True,
    )

    for batch in data_iterator:
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(openai_client, MODEL_NAME, bi["prompt"], bi["row"], batch.step, batch.epoch)
                    for _ in range(NUM_GENERATIONS)
                )
                for bi in batch.items
            )
        )

        # 5) 过滤无效轨迹（例如生成失败/类型异常等），并仅保留至少包含 2 条轨迹的 group
        valid_train_groups = []
        for group in train_groups:
            valid_trajectories = [traj for traj in group if isinstance(traj, art.Trajectory)]
            if len(valid_trajectories) > 1:
                valid_train_groups.append(art.TrajectoryGroup(valid_trajectories))

        # === 对本 step 训练轨迹做聚合并打点 ===
        all_train_trajs = [traj for g in valid_train_groups for traj in g]  # 展平
        if all_train_trajs:
            rewards = [float(t.reward) for t in all_train_trajs]
            rms = [float(t.metrics.get("rm", 0.0)) for t in all_train_trajs]
            matches = [int(t.metrics.get("matches", 0)) for t in all_train_trajs]
            lengths = [int(t.metrics.get("length", 0)) for t in all_train_trajs]

            log_dict = {
                "global_step": int(batch.step),
                "train/reward/mean": sum(rewards) / len(rewards),
                "train/reward/max": max(rewards),
                "train/reward/min": min(rewards),
                "train/rm/mean": sum(rms) / len(rms),
                "train/match_rate": sum(matches) / len(matches),
                "train/length/mean": sum(lengths) / len(lengths) if lengths else 0,
                "train/num_groups": len(valid_train_groups),
                "train/num_trajs": len(all_train_trajs),
                # 直方图（在 UI 里看分布）
                "train/reward/hist": wandb.Histogram(rewards),
                "train/rm/hist": wandb.Histogram(rms),
            }
            wandb.log(log_dict)

        if not valid_train_groups:
            print(f"警告：第 {batch.step} 步未生成有效轨迹，跳过参数更新。")
            continue

        # 6) 用有效轨迹更新模型参数（策略改进）
        await model.train(valid_train_groups, config=art.TrainConfig(learning_rate=LEARNING_RATE))

        # === 可选：每隔一段步数记录几个样例到表格（避免过频打点）
        if batch.step % max(10, EVAL_STEPS) == 0 and all_train_trajs:
            table = wandb.Table(columns=["step", "epoch", "title", "rm", "match", "reward"])
            # 取前 8 个样例
            for t in all_train_trajs[:8]:
                table.add_data(
                    int(batch.step),
                    int(batch.epoch),
                    # 取消息里最后一个 assistant 的内容（已在 rollout 里放在 choice.message）
                    getattr(t.messages_and_choices[-1].message, "content", "") if hasattr(t.messages_and_choices[-1], "message") else "",
                    float(t.metrics.get("rm", 0.0)),
                    int(t.metrics.get("matches", 0)),
                    float(t.reward),
                )
            wandb.log({"global_step": int(batch.step), "train/examples": table})

        # 7) 周期性评估
        if batch.step > 0 and batch.step % EVAL_STEPS == 0:
            print(f"\n--- 在第 {batch.step} 步进行评估 ---")
            val_trajectories = await art.gather_trajectories(
                (
                    rollout(openai_client, MODEL_NAME, item["prompt"], item["row"], batch.step, batch.epoch)
                    for item in val_data_list
                ),
                pbar_desc="val",
            )
            await model.log(val_trajectories)
            await model.delete_checkpoints()

            # === 验证集聚合与打点 ===
            val_trajs = [t for t in val_trajectories if isinstance(t, art.Trajectory)]
            if val_trajs:
                v_rewards = [float(t.reward) for t in val_trajs]
                v_rms = [float(t.metrics.get("rm", 0.0)) for t in val_trajs]
                v_matches = [int(t.metrics.get("matches", 0)) for t in val_trajs]
                v_lengths = [int(t.metrics.get("length", 0)) for t in val_trajs]
                wandb.log(
                    {
                        "global_step": int(batch.step),
                        "val/reward/mean": sum(v_rewards) / len(v_rewards),
                        "val/reward/max": max(v_rewards),
                        "val/reward/min": min(v_rewards),
                        "val/rm/mean": sum(v_rms) / len(v_rms),
                        "val/match_rate": sum(v_matches) / len(v_matches),
                        "val/length/mean": sum(v_lengths) / len(v_lengths) if v_lengths else 0,
                        "val/num_trajs": len(val_trajs),
                        "val/reward/hist": wandb.Histogram(v_rewards),
                        "val/rm/hist": wandb.Histogram(v_rms),
                    }
                )

    print("训练完成。")
    # === 新增：优雅收尾 ===
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
