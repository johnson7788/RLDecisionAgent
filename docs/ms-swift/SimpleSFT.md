# 简单微调
https://swift.readthedocs.io/zh-cn/latest/GetStarted/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.html

swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot



[INFO:modelscope] Download model 'Qwen/Qwen2.5-7B-Instruct' successfully.
[INFO:modelscope] Target directory already exists, skipping creation.
[INFO:swift] Loading the model using model_dir: /mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct
[INFO:swift] model_kwargs: {'device_map': 'cuda:0'}
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.23it/s]
[INFO:swift] model.hf_device_map: {'': device(type='cuda', index=0)}
[INFO:swift] model_info: ModelInfo(model_type='qwen2_5', model_dir='/mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct', torch_dtype=torch.bfloat16, max_model_len=32768, quant_method=None, quant_bits=None, rope_scaling=None, is_moe_model=False, config=Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "pad_token_id": 151643,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "transformers_version": "4.56.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}
, task_type='causal_lm', num_labels=None)
[INFO:swift] model.generation_config: GenerationConfig {
  "bos_token_id": 151643,
  "eos_token_id": [
    151645,
    151643
  ],
  "max_new_tokens": 64,
  "pad_token_id": 151643,
  "repetition_penalty": 1.05
}
[INFO:swift] default_system: 'You are a helpful assistant.'
[INFO:swift] max_length: 2048
[INFO:swift] response_prefix: ''
[INFO:swift] agent_template: hermes
[INFO:swift] Start time of running main: 2025-09-26 12:58:24.591065
[INFO:swift] swift.__version__: 3.8.1
[INFO:swift] SelfCognitionPreprocessor has been successfully configured with name: ('swift-robot', 'swift-robot'), author: ('swift', 'swift').
[INFO:swift] Downloading the dataset from ModelScope, dataset_id: AI-ModelScope/alpaca-gpt4-data-zh
Downloading [README.md]: 100%|█████████████████████████████████████████████████████████████████████████████| 1.23k/1.23k [00:00<00:00, 7.09MB/s]
Downloading data: 31.8MB [00:08, 3.66MB/s]
Generating train split: 48818 examples [00:00, 65753.46 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 48818/48818 [00:01<00:00, 34896.75 examples/s]
[INFO:swift] Downloading the dataset from ModelScope, dataset_id: AI-ModelScope/alpaca-gpt4-data-en
Downloading [README.md]: 100%|██████████████████████████████████████████████████████████████████████████████| 1.23k/1.23k [00:00<00:00, 803kB/s]
Downloading data: 88.3MB [00:24, 3.62MB/s]
Generating train split: 52002 examples [00:01, 42855.58 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 52002/52002 [00:01<00:00, 40120.21 examples/s]
[INFO:swift] Downloading the dataset from ModelScope, dataset_id: swift/self-cognition
Downloading [README.md]: 100%|█████████████████████████████████████████████████████████████████████████████| 1.98k/1.98k [00:00<00:00, 14.8MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████| 23.8k/23.8k [00:00<00:00, 44.1MB/s]
Generating train split: 108 examples [00:00, 27537.07 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 108/108 [00:00<00:00, 4136.58 examples/s]
[WARNING:swift] dataset_sample:500 is greater than len(dataset):108, repeated sampling will be performed.
[INFO:swift] train_dataset: Dataset({
    features: ['messages'],
    num_rows: 1500
})
[INFO:swift] val_dataset: None
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:01<00:00, 1026.76 examples/s]
[INFO:swift] [INPUT_IDS] [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 105043, 9686, 70, 417, 101037, 151645, 198, 151644, 77091, 198, 99520, 9370, 3837, 111020, 15672, 38, 2828, 1773, 104198, 67071, 70642, 100013, 100623, 48692, 100168, 104949, 70642, 12, 18247, 1773, 107055, 99885, 106603, 57191, 85106, 100364, 3837, 100437, 102422, 69041, 35946, 107666, 1773, 151645]
[INFO:swift] [INPUT] <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你是chatgpt吗<|im_end|>
<|im_start|>assistant
不是的，我不是ChatGPT。我是由swift开发的人工智能模型swift-robot。如果有任何疑问或需要帮助，欢迎随时向我提问。<|im_end|>
[INFO:swift] [LABELS_IDS] [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 99520, 9370, 3837, 111020, 15672, 38, 2828, 1773, 104198, 67071, 70642, 100013, 100623, 48692, 100168, 104949, 70642, 12, 18247, 1773, 107055, 99885, 106603, 57191, 85106, 100364, 3837, 100437, 102422, 69041, 35946, 107666, 1773, 151645]
[INFO:swift] [LABELS] [-100 * 24]不是的，我不是ChatGPT。我是由swift开发的人工智能模型swift-robot。如果有任何疑问或需要帮助，欢迎随时向我提问。<|im_end|>
[INFO:swift] Dataset Token Length: 130.807333±103.982221, min=29.000000, max=550.000000, size=1500
[INFO:swift] The TrainArguments will be saved in: /workspace/verl/output/v0-20250926-114832/args.json
/usr/local/lib/python3.11/site-packages/awq/__init__.py:21: DeprecationWarning:
I have left this message as the final dev message to help you transition.
Important Notice:
- AutoAWQ is officially deprecated and will no longer be maintained.
- The last tested configuration used Torch 2.6.0 and Transformers 4.51.3.
- If future versions of Transformers break AutoAWQ compatibility, please report the issue to the Transformers project.

Alternative:
- AutoAWQ has been adopted by the vLLM Project: https://github.com/vllm-project/llm-compressor

For further inquiries, feel free to reach out:
- X: https://x.com/casper_hansen_
- LinkedIn: https://www.linkedin.com/in/casper-hansen-804005170/

  warnings.warn(_FINAL_DEV_MESSAGE, category=DeprecationWarning, stacklevel=1)
[INFO:swift] lora_config: LoraConfig(task_type='CAUSAL_LM', peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='/mnt/
workspace/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct', revision=None, inference_mode=False, r=8, target_modules={'down_proj', 'o_pr
oj', 'gate_proj', 'q_proj', 'v_proj', 'up_proj', 'k_proj'}, exclude_modules=None, lora_alpha=32, lora_dropout=0.05, fan_in_fan_out=False, bias='
none', use_rslora=False, modules_to_save=[], init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_patte
rn={}, megatron_config=None, megatron_core='megatron.core', trainable_token_indices=None, loftq_config={}, eva_config=None, corda_config=None, u
se_dora=False, use_qalora=False, qalora_group_size=16, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lo
ra_bias=False, target_parameters=None, lora_dtype=None, lorap_lr_ratio=None, lorap_emb_lr=1e-06)
[INFO:swift] model: PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): Qwen2ForCausalLM(
      (model): Qwen2Model(
        (embed_tokens): Embedding(152064, 3584)
        (layers): ModuleList(
          (0-27): 28 x Qwen2DecoderLayer(
            (self_attn): Qwen2Attention(
-               (q_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=3584, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3584, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=512, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=512, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=512, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=512, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=3584, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3584, bias=False)
              (down_proj): lora.Linear(
                (base_layer): Linear(in_features=18944, out_features=3584, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=18944, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3584, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
            (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
          )
        )
        (norm): Qwen2RMSNorm((3584,), eps=1e-06)
        (rotary_emb): Qwen2RotaryEmbedding()
      )
      (lm_head): Linear(in_features=3584, out_features=152064, bias=False)
    )
  )
)
[INFO:swift] model_parameter_info: PeftModelForCausalLM: 7635.8016M Params (20.1851M Trainable [0.2643%]), 0.0001M Buffers.
/usr/local/lib/python3.11/site-packages/swift/trainers/mixin.py:104: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
  super().__init__(
/usr/local/lib/python3.11/site-packages/deepspeed/ops/op_builder/builder.py:16: DeprecationWarning: The distutils package is deprecated and slated for removal in Python 3.12. Use setuptools or check PEP 632 for potential alternatives
  import distutils.ccompiler
/usr/local/lib/python3.11/site-packages/deepspeed/ops/op_builder/builder.py:18: DeprecationWarning: The distutils.sysconfig module is deprecated, use sysconfig instead
  import distutils.sysconfig
[2025-09-26 13:00:44,888] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to cuda (auto detect)
df: /root/.triton/autotune: 没有那个文件或目录
[2025-09-26 13:00:46,797] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[INFO:swift] use_reentrant: True
[INFO:swift] The logging file will be saved in: /workspace/verl/output/v0-20250926-114832/logging.jsonl
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None}.
Train:   0%|                                                                                                             | 0/94 [00:00<?, ?it/s]/usr/local/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[INFO:swift] use_logits_to_keep: True
{'loss': 1.4749825, 'grad_norm': 1.59615517, 'learning_rate': 2e-05, 'token_acc': 0.66827957, 'epoch': 0.01, 'global_step/max_steps': '1/94', 'percentage': '1.06%', 'elapsed_time': '7s', 'remaining_time': '12m 6s', 'memory(GiB)': 16.75, 'train_speed(iter/s)': 0.128088}
{'loss': 1.72185874, 'grad_norm': 1.92592478, 'learning_rate': 0.0001, 'token_acc': 0.64476276, 'epoch': 0.05, 'global_step/max_steps': '5/94', 'percentage': '5.32%', 'elapsed_time': '32s', 'remaining_time': '9m 36s', 'memory(GiB)': 17.47, 'train_speed(iter/s)': 0.154297}
{'loss': 1.35411816, 'grad_norm': 0.95349097, 'learning_rate': 9.922e-05, 'token_acc': 0.6687643, 'epoch': 0.11, 'global_step/max_steps': '10/94', 'percentage': '10.64%', 'elapsed_time': '1m 3s', 'remaining_time': '8m 50s', 'memory(GiB)': 17.47, 'train_speed(iter/s)': 0.158366}
{'loss': 1.27653036, 'grad_norm': 0.64977044, 'learning_rate': 9.692e-05, 'token_acc': 0.66162008, 'epoch': 0.16, 'global_step/max_steps': '15/94', 'percentage': '15.96%', 'elapsed_time': '1m 33s', 'remaining_time': '8m 14s', 'memory(GiB)': 18.33, 'train_speed(iter/s)': 0.159901}
{'loss': 1.22438469, 'grad_norm': 0.62878072, 'learning_rate': 9.315e-05, 'token_acc': 0.66291977, 'epoch': 0.21, 'global_step/max_steps': '20/94', 'percentage': '21.28%', 'elapsed_time': '2m 4s', 'remaining_time': '7m 41s', 'memory(GiB)': 18.33, 'train_speed(iter/s)': 0.160312}
