# 错误收集
## 1. 显卡
Tesla V100-SXM2-32GB 不支持
flash_attn_gpu.varlen_fwd(
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: FlashAttention only supports Ampere GPUs or newer.

## 2. 代理
env_file配置代理，然后pycharm中的Paths to env加载env文件，即可使用代理
```
ALL_PROXY=http://127.0.0.1:7890
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```

## 3. SFT错误
```
 bash run_qwen2-05b_sft.sh
+ export CUDA_VISIBLE_DEVICES=1,2
+ CUDA_VISIBLE_DEVICES=1,2
+ nnodes=1
+ nproc_per_node=2
+ experiment_name=multiturn-sft-Qwen2.5-0.5B-Instruct
+ HDFS_ROOT=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool
+ DATA_ROOT=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool
+ TRAIN_DATA=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
+ EVAL_DATA=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
+ MODEL_PATH=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/model/Qwen2.5-0.5B-Instruct
+ SAVE_PATH=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct
+ torchrun --nnodes=1 --nproc_per_node=2 -m verl.trainer.fsdp_sft_trainer data.train_files=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet data.val_files=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet data.max_length=16384 data.train_batch_size=32 data.multiturn.enable=true data.multiturn.messages_key=messages data.multiturn.tools_key=tools data.micro_batch_size_per_gpu=4 model.partial_pretrain=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/model/Qwen2.5-0.5B-Instruct model.strategy=fsdp trainer.default_local_dir=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct trainer.project_name=wuxibin-multiturn-sft trainer.experiment_name=multiturn-sft-Qwen2.5-0.5B-Instruct 'trainer.logger=["console","wandb"]' trainer.total_epochs=6 ulysses_sequence_parallel_size=2 use_remove_padding=true
W0731 10:49:55.044000 3247964 /media/wac/backup/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/run.py:792]
W0731 10:49:55.044000 3247964 /media/wac/backup/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/run.py:792] *****************************************
W0731 10:49:55.044000 3247964 /media/wac/backup/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0731 10:49:55.044000 3247964 /media/wac/backup/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/run.py:792] *****************************************
Normalize batch size by dp 1
Using sequence parallel size: 2
Using remove padding: True
Using SP rank 0 and size 1 for data distribution
Each SP rank gets different data, but the same data WITHIN the same rank
Using FSDP rank 0 and size 1 for data distribution
Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Monkey patch _flash_attention_forward in transformers.integrations.flash_attentionMonkey patch _flash_attention_forward in transformers.integrations.flash_attention

Skipping monkey patch for Qwen2ForCausalLM as use_fused_kernels is False or fused_kernels_backend is None
Skipping monkey patch for Qwen2ForCausalLM as use_fused_kernels is False or fused_kernels_backend is None
functools.partial(<function _or_policy at 0x748db7acd630>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x748db7acd510>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
NCCL version 2.21.5+cuda12.4
Number of steps/epoch 0, number of epochs 6, total number of steps 0
Error executing job with overrides: ['data.train_files=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet', 'data.val_files=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet', 'data.max_length=16384', 'data.train_batch_size=32', 'data.multiturn.enable=true', 'data.multiturn.messages_key=messages', 'data.multiturn.tools_key=tools', 'data.micro_batch_size_per_gpu=4', 'model.partial_pretrain=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/model/Qwen2.5-0.5B-Instruct', 'model.strategy=fsdp', 'trainer.default_local_dir=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct', 'trainer.project_name=wuxibin-multiturn-sft', 'trainer.experiment_name=multiturn-sft-Qwen2.5-0.5B-Instruct', 'trainer.logger=["console","wandb"]', 'trainer.total_epochs=6', 'ulysses_sequence_parallel_size=2', 'use_remove_padding=true']
{'data': {'train_batch_size': 32, 'micro_batch_size': None, 'micro_batch_size_per_gpu': 4, 'train_files': '/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet', 'val_files': '/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet', 'prompt_key': 'question', 'response_key': 'answer', 'prompt_dict_keys': None, 'response_dict_keys': None, 'multiturn': {'enable': True, 'messages_key': 'messages', 'tools_key': 'tools', 'enable_thinking_key': 'enable_thinking'}, 'max_length': 16384, 'truncation': 'error', 'balance_dp_token': False, 'chat_template': None, 'custom_cls': {'path': None, 'name': None}, 'use_shm': False}, 'model': {'partial_pretrain': '/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/model/Qwen2.5-0.5B-Instruct', 'use_shm': False, 'fsdp_config': {'model_dtype': 'fp32', 'wrap_policy': {'min_num_params': 0}, 'cpu_offload': False, 'offload_params': False}, 'external_lib': None, 'enable_gradient_checkpointing': True, 'trust_remote_code': False, 'lora_rank': 0, 'lora_alpha': 16, 'target_modules': 'all-linear', 'use_liger': False, 'strategy': 'fsdp'}, 'optim': {'lr': 1e-05, 'betas': [0.9, 0.95], 'weight_decay': 0.01, 'warmup_steps_ratio': 0.1, 'clip_grad': 1.0, 'lr_scheduler': 'cosine'}, 'ulysses_sequence_parallel_size': 2, 'use_remove_padding': True, 'trainer': {'default_local_dir': '/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct', 'default_hdfs_dir': None, 'project_name': 'wuxibin-multiturn-sft', 'experiment_name': 'multiturn-sft-Qwen2.5-0.5B-Instruct', 'total_epochs': 6, 'total_training_steps': None, 'logger': ['console', 'wandb'], 'seed': 1, 'save_freq': -1, 'test_freq': -1, 'nnodes': 1, 'n_gpus_per_node': 8, 'max_ckpt_to_keep': None, 'resume_mode': 'auto', 'resume_from_path': None, 'checkpoint': {'save_contents': ['model', 'optimizer', 'extra'], 'load_contents': '${trainer.checkpoint.save_contents}'}, 'device': 'cuda'}}
Traceback (most recent call last):
  File "/media/wac/backup/john/johnson/RLDecisionAgent/verl/verl/trainer/fsdp_sft_trainer.py", line 801, in main
    run_sft(config)
  File "/media/wac/backup/john/johnson/RLDecisionAgent/verl/verl/trainer/fsdp_sft_trainer.py", line 794, in run_sft
    trainer.fit()
  File "/media/wac/backup/john/johnson/RLDecisionAgent/verl/verl/trainer/fsdp_sft_trainer.py", line 716, in fit
    start_epoch = global_step // self.steps_per_epoch
ZeroDivisionError: integer division or modulo by zero

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[rank1]:[W731 10:50:06.313201823 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0731 10:50:07.575000 3247964 /media/wac/backup/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3248029 closing signal SIGTERM
E0731 10:50:08.043000 3247964 /media/wac/backup/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 1 (pid: 3248030) of binary: /home/wac/johnson/anaconda3/envs/gpt/bin/python
Traceback (most recent call last):
  File "/home/wac/johnson/anaconda3/envs/gpt/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
verl.trainer.fsdp_sft_trainer FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-07-31_10:50:07
  host      : yaqiyun-SYS-4028GR-TR2
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 3248030)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```