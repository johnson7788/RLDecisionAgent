# 标题生成的训练任务， 对比不同的训练方法
给hacker news生成新闻标题。

## 使用trl框架训练GRPO
[reference_grpo_trainer.py](reference_grpo_trainer.py)

## 使用ART框架训练GRPO
[train.py](train.py)

# 处理提示词，获取数据等
utils.py


# 训练脚本 (`reference_grpo_trainer.py` 和 `train.py`) 的核心差别在于 **训练框架、训练逻辑、模型调用方式**，以及 **奖励函数的实现方式**。
---

## 1. 使用的训练框架

* **`reference_grpo_trainer.py`**

  * 使用 **TRL 库的 `GRPOTrainer`**（Group Relative Policy Optimization，一种RLHF/RLAIF强化学习训练器）。
  * 训练过程是“标准化”的：配置超参数 → 加载模型（Unsloth 加速 + LoRA 适配器）→ 调用 `GRPOTrainer.train()`。

* **`train.py`**

  * 使用 **ART  + OpenPipe** 的自定义训练框架。
  * 训练循环是 **手写的**：手动 rollout → reward → trajectory → `model.train()`。

🔑 区别：前者是**库提供的高阶封装**，后者是**自定义控制更灵活**的训练循环。

---

## 2. 模型调用方式

* **`reference_grpo_trainer.py`**

  * 直接在本地加载 `Qwen/Qwen2.5-0.5B-Instruct` 模型。
  * 用 **Unsloth + VLLM** 加速推理 (`FastLanguageModel.fast_generate`)。
  * LoRA 参数合并到本地模型后进行训练。

* **`train.py`**

  * 定义了一个 **ART TrainableModel**，用 OpenAI/ART API 的方式来调用模型。
  * 生成和验证标题时调用 `openai.AsyncOpenAI` 客户端。
  * 推理走 **OpenAI/ART API**，而不是直接用本地 HuggingFace 模型。

🔑 区别：前者完全本地跑，后者基于 API 异步调用（可分布式、可观测性更强）。

---

## 3. 奖励函数实现

* **`reference_grpo_trainer.py`**

  * 在 `reward_func` 里同步调用 `calculate_rewards`。
  * 奖励由两个部分组成：

    1. **Reward Model (RM)** 分数（调用 `score_title` 服务）。
    2. **标题与正文匹配验证**（用同一个 Qwen 模型快速判断 True/False）。
  * 最终 reward = RM 分数 \* 是否匹配。

* **`train.py`**

  * `rollout` 阶段生成标题 → 调用 `check_title_matches_body`（用 OpenAI API 调用基模型判断 True/False）。
  * 然后再请求 **外部 Reward Model API** (`score_title`)。
  * reward 同样是匹配判定后才保留 RM 分数，否则置 0。

🔑 区别：**Reference 用本地模型做匹配判断**，**Train 用 API 做匹配判断**。后者更一致但更依赖外部服务。

---

## 4. 训练循环与验证

* **`reference_grpo_trainer.py`**

  * 内置 **ValidationCallback**，定期保存 LoRA 权重并做验证生成。
  * 验证集 rollouts + reward 计算在 callback 内完成。

* **`train.py`**

  * 自己写的训练 loop：

    * `for batch in data_iterator` → 多次 rollout → 过滤有效 trajectories → 调用 `model.train()`。
    * 定期手动触发验证 (`if batch.step % EVAL_STEPS == 0`)。
  * 验证时直接 rollout 全部 val 数据 → `model.log()`。

🔑 区别：Reference 脚本训练-验证是**trainer 框架自动化**，Train 脚本是**显式写循环**。

---

## 5. 数据与预处理

* **两者相同点**

  * 数据源都来自 HuggingFace `OpenPipe/hacker-news-scraped-stories-filtered`。
  * 都会过滤过长的样本（token 长度 > 8192 时丢弃）。
  * Prompt 格式一致：system 指令 + user 提供正文。

* **细节差异**

  * `reference_grpo_trainer.py`：过滤函数 `filter_on_length` 是基于 `PreTrainedTokenizer`。
  * `train.py`：用 `AutoTokenizer`，加了异常处理，容错性更强。

---

## 总结

* **`reference_grpo_trainer.py` = HuggingFace TRL + GRPOTrainer 的标准化实现**

  * 适合快速试验和复现论文。
  * 本地模型推理 & LoRA 高效训练。
  * 框架帮你管理训练 loop、日志、checkpoint。

* **`train.py` = ART/OpenPipe 的自定义训练 loop**

  * 更灵活，可以控制 rollout、reward 计算和日志上报。
  * 支持异步 API 调用，适合大规模分布式实验。
  * 训练逻辑由用户完全掌控，但实现更复杂。


# Train.py的伪代码
load train/val datasets → map(scraped_body → chat messages) → length filter
init ART backend & TrainableModel (LoRA, grad clip, etc.)
get start_step
for each epoch/step batch:
  for each item in batch:
    generate NUM_GENERATIONS titles with trainable model (logprobs on)
    for each title:
      check match = validator(base_model) → True/False
      rm = reward_model_score(HTTP)
      reward = 0 if not match else rm
      build trajectory(messages+choice, reward, metrics)
    group trajectories per item
  filter groups with ≥2 valid trajectories
  if any:
    model.train(groups, lr)
  if step % EVAL_STEPS == 0:
    val_trajectories = rollout on all val items
    model.log(val_trajectories); model.delete_checkpoints()


# 训练
1. 启动本地奖励模型
python local_reward.py

2. 测试本地奖励模型是否OK
python reward_local_test.py

3. 在.env中加上
REWARD_MODEL_URL=http://127.0.0.1:7000/score

4. 测试数据加载是否正常
python check_dataset.py

5. 启动训练
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
python train.py

6. 进行测试