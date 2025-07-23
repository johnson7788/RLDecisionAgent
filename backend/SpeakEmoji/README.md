# Emoji翻译机

使用SFT和强化学习进行训练

输入示例:
Please convert the string "h-e-l-l-o-1-2" to emojis.

推理：
h → 🕳️  
e → 🐘  
l → 🦁  
l → 🦁  
o → 🐙  
1 → 1️⃣  
2 → 2️⃣

输出：
Final emoji string: \boxed{🕳️🐘🦁🦁🐙1️⃣2️⃣


# 文件
| 文件                              | 说明                                  |
| ------------------------------- | ----------------------------------- |
| `create_dataset.py`             | 自动生成训练/验证数据（SFT 和 RL）               |
| `reward_function.py`            | 比较模型输出的 emoji 序列与 ground truth 是否一致 |
| `train_sft.sh`, `train_grpo.sh` | 基本结构无需更改，仅改路径和 reward 名             |


# 解释train_grpo.sh
```
python3 -m verl.trainer.main_ppo \
启动 Verl 中的 main_ppo 模块进行训练

algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
使用 GRPO 来估计 Advantage（而非标准 GAE）。
训练时 不把 KL 散度惩罚项加入 reward，完全靠 reward_function.py 里的 reward。

指定训练和验证文件，格式为 .parquet。
train_batch_size=128：训练时每个 batch 含 128 条数据。
truncation='error'：若 prompt/response 超出长度就报错。

data.train_files=$HOME/data/speek_emoji/rl/train.parquet
data.val_files=$HOME/data/speek_emoji/rl/test.parquet
data.train_batch_size=128
data.max_prompt_length=128
data.max_response_length=128
data.filter_overlong_prompts=False
data.truncation='error'

初始化策略模型（Actor）和参考模型（Ref）使用 SFT 模型的第 105 步检查点。
actor_rollout_ref.model.path=./models/sft/global_step_105


优化器学习率 1e-6。
动态 batch size，最大 token 总数限制为 5000。
use_remove_padding=True：模型 forward 时去除多余 padding。
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.actor.ppo_mini_batch_size=16
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5000


actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.0
actor_rollout_ref.actor.kl_loss_type=low_var_kl
不使用 KL loss，kl_loss_coef=0.0。
kl_loss_type=low_var_kl 只是占位（无效，因 kl loss 被关了）。


actor_rollout_ref.actor.entropy_coeff=0
不加 entropy regularization（不鼓励探索）。



actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
使用 FSDP（Fully Sharded Data Parallel）节省显存。
启用梯度检查点技术（节省显存，但训练稍慢）。


actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
actor_rollout_ref.rollout.n=8
使用 vllm 推理引擎（高性能生成）。
每轮 rollout 生成 8 个样本。
限制推理使用显存最多 70%。


trainer.critic_warmup=0
trainer.logger='["console","tensorboard"]'
trainer.project_name='verl_example'
trainer.experiment_name='smol135m_grpo'
trainer.val_before_train=True
trainer.n_gpus_per_node=1
trainer.nnodes=1
trainer.save_freq=-1
trainer.test_freq=5
trainer.total_epochs=2
critic_warmup=0：Critic 和 Actor 一起训练。
输出日志到控制台和 TensorBoard。
总训练 2 个 epoch。
每 5 轮测试一次。
save_freq=-1：用默认策略保存模型。

custom_reward_function.path=reward_function.py
custom_reward_function.name=char_to_emoji_reward_function
自定义奖励函数在当前目录的 reward_function.py 文件中。
使用其中的 char_to_emoji_reward_function 函数，比如根据输出内容计算 reward。

```