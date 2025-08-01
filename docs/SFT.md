# SFT支持的参数
verl/verl/verl/trainer# python fsdp_sft_trainer.py -h
fsdp_sft_trainer is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

actor: actor, dp_actor, megatron_actor
critic: critic, dp_critic, megatron_critic
data: legacy_data
npu_profile: npu_profile
ref: dp_ref, megatron_ref, ref
reward_model: dp_reward_model, megatron_reward_model, reward_model
rollout: rollout


== Config ==
Override anything in the config (foo.bar=value)

data:
  train_batch_size: 256
  micro_batch_size: null
  micro_batch_size_per_gpu: 4
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet
  prompt_key: question
  response_key: answer
  prompt_dict_keys: null
  response_dict_keys: null
  multiturn:
    enable: false
    messages_key: messages
    tools_key: tools
    enable_thinking_key: enable_thinking
  max_length: 1024
  truncation: error
  balance_dp_token: false
  chat_template: null
  custom_cls:
    path: null
    name: null
  use_shm: false
model:
  partial_pretrain: ~/models/gemma-1.1-7b-it
  use_shm: false
  fsdp_config:
    model_dtype: fp32
    wrap_policy:
      min_num_params: 0
    cpu_offload: false
    offload_params: false
  external_lib: null
  enable_gradient_checkpointing: true
  trust_remote_code: false
  lora_rank: 0
  lora_alpha: 16
  target_modules: all-linear
  use_liger: false
  strategy: fsdp2
optim:
  lr: 1.0e-05
  betas:
  - 0.9
  - 0.95
  weight_decay: 0.01
  warmup_steps_ratio: 0.1
  clip_grad: 1.0
  lr_scheduler: cosine
ulysses_sequence_parallel_size: 1
use_remove_padding: false
trainer:
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
  default_hdfs_dir: null
  project_name: gsm8k-sft
  experiment_name: test
  total_epochs: 4
  total_training_steps: null
  logger:
  - console
  - wandb
  seed: 1
  save_freq: -1
  test_freq: -1
  nnodes: 1
  n_gpus_per_node: 8
  max_ckpt_to_keep: null
  resume_mode: auto
  resume_from_path: null
  checkpoint:
    save_contents:
    - model
    - optimizer
    - extra
    load_contents: ${trainer.checkpoint.save_contents}
  device: cuda


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help