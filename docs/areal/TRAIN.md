# 训练参数
```
exp_ctrl:
  total_train_epochs: 5
  save_freq_epochs: 1
  ckpt_freq_secs: 600
torch_cache_mysophobia: true
recover_mode: auto
recover_retries: 10
```

total_train_epochs ：要运行的训练周期数
save_freq_epochs ：模型在各个时期的保存频率
ckpt_freq_secs ：检查点频率（以秒为单位）
torch_cache_mysophobia ：积极清除 PyTorch 缓存
recover_mode ：失败时自动实验恢复


# PPO 特定参数
```
ppo:
  gen:
    max_new_tokens: 27648
    min_new_tokens: 0
    top_p: 1.0
    temperature: 1.0
  ppo_n_minibatches: 4
  kl_ctl: 0.0
  discount: 1.0
  disable_value: true
  reward_output_scaling: 5
  adv_norm: true
```
gen ：策略采样的生成参数
ppo_n_minibatches ：PPO 更新的小批次数量
kl_ctl ：KL 散度惩罚系数
disable_value ：禁用价值函数训练
reward_output_scaling ：缩放奖励信号

# 内存管理参数
```
actor_train:
  mb_spec:
    max_tokens_per_mb: 30720

actor:
  sglang:
    mem_fraction_static: 0.8
    triton_attention_num_kv_splits: 16
```
max_tokens_per_mb ：用于内存控制的每个微批次的最大令牌数
mem_fraction_static ：SGLang 的静态内存
triton_attention_num_kv_splits ：注意内核优化


# 训练
python3 -m areal.launcher.local examples/lite/gsm8k_grpo.py --config examples/lite/configs/gsm8k_grpo.yaml
