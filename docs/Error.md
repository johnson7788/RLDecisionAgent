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

## 3. SFT错误, 数据集过少，检查数据集数量
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

python，检查数据数量是否过少
import pandas as pd
df = pd.read_parquet('/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet')
print(len(df))
print(df.head())

```



## 4. Tools的Json报错, 经过Debug，发现backend/reTool/retool_sft_preprocess.py中的tools没有转换成json，还是字符串，所以加载时有问题 
record[messages_key] 和 record[tools_key] 从 Pandas DataFrame 中读取数据时，如果原始 Parquet 文件中的列存储为结构化数组，Pandas 可能会将其解释为 numpy.ndarray 或 numpy.void 类型。
isinstance(obj, np.ndarray) 检查对象是否为 NumPy 数组。
.tolist() 方法可以将 NumPy 数组转换为标准的 Python 列表。
numpy.void.item() 可以将 NumPy 的结构化数据类型（类似于 C 结构体）转换为 Python 字典。
通过递归或逐层检查和转换，确保传递给 apply_chat_template 的 messages 和 tools 参数完全由 Python 原生类型（列表、字典、字符串、数字）组成。

```
{'content': "Counter({'P': 3, 'I': 3, 'E': 2, 'N': 2, 'G': 1, 'O': 1, 'R': 1, 'C': 1, 'L': 1})", 'role': 'tool', 'tool_calls': None}
 {'content': "The output shows the letters with their counts: P (3), I (3), E (2), N (2), and single occurrences for G, O, R, C, L. Now, we need to determine the maximum number of draws required to guarantee at least two pairs.\n\nUsing the pigeonhole principle, the worst-case scenario involves drawing all single-occurrence letters first (5 letters). For the remaining letters (P, I, E, N), we maximize draws without forming two pairs. The optimal strategy is to take 3 letters from one multi-occurrence group (e.g., 3 P's creates 1 pair + 1 single) and 1 each from the others. This gives:\n\nSingle letters: 5\nMulti-occurrence letters: 3 (P) + 1 (I) + 1 (E) + 1 (N) = 6\nTotal: 5 + 6 = 11\n\nThe next draw (12th) must complete a second pair. Thus, the answer is:\n\n\n\\boxed{12}", 'role': 'assistant', 'tool_calls': None}]
Tools: [{"type": "function", "function": {"name": "code_interpreter", "description": "A tool for executing code.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to execute."}}, "required": ["code"]}}}]
Enable thinking: None
Error executing job with overrides: ['data.train_files=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet', 'data.val_files=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet', 'data.max_length=16384', 'data.train_batch_size=32', 'data.multiturn.enable=true', 'data.multiturn.messages_key=messages', 'data.multiturn.tools_key=tools', 'data.micro_batch_size_per_gpu=4', 'model.partial_pretrain=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/model/Qwen2.5-0.5B-Instruct', 'model.strategy=fsdp', 'trainer.default_local_dir=/media/wac/backup/john/johnson/RLDecisionAgent/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct', 'trainer.project_name=wuxibin-multiturn-sft', 'trainer.experiment_name=multiturn-sft-Qwen2.5-0.5B-Instruct', 'trainer.logger=["console","wandb"]', 'trainer.total_epochs=6', 'ulysses_sequence_parallel_size=2', 'use_remove_padding=true']
Traceback (most recent call last):
  File "/media/wac/backup/john/johnson/RLDecisionAgent/verl/verl/trainer/fsdp_sft_trainer.py", line 801, in main
    run_sft(config)
  File "/media/wac/backup/john/johnson/RLDecisionAgent/verl/verl/trainer/fsdp_sft_trainer.py", line 794, in run_sft
    trainer.fit()
  File "/media/wac/backup/john/johnson/RLDecisionAgent/verl/verl/trainer/fsdp_sft_trainer.py", line 721, in fit
    for step_in_epoch, data in enumerate(
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/tqdm/std.py", line 1169, in __iter__
    for obj in iterable:
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torchdata/stateful_dataloader/stateful_dataloader.py", line 450, in __next__
    return super().__next__()
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torchdata/stateful_dataloader/stateful_dataloader.py", line 1456, in _next_data
    return self._process_data(data, worker_id, state_dict)
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torchdata/stateful_dataloader/stateful_dataloader.py", line 1543, in _process_data
    data.reraise()
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/_utils.py", line 733, in reraise
    raise exception
ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torchdata/stateful_dataloader/worker.py", line 242, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[union-attr]
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/media/wac/backup/john/johnson/RLDecisionAgent/verl/verl/utils/dataset/multiturn_sft_dataset.py", line 234, in __getitem__
    full_tokens = tokenizer.apply_chat_template(
  File "/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py", line 1665, in apply_chat_template
    raise ValueError(
ValueError: Tools should either be a JSON schema, or a callable function with type hints and a docstring suitable for auto-conversion to a schema.


Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[rank1]:[W731 11:08:39.927063060 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: W0731 11:08:41.198000 3253423 /media/wac/backup/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3253490 closing signal SIGTERM
E0731 11:08:41.515000 3253423 /media/wac/backup/anaconda3/envs/gpt/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 1 (pid: 3253491) of binary: /home/wac/johnson/anaconda3/envs/gpt/bin/python
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
  time      : 2025-07-31_11:08:41
  host      : yaqiyun-SYS-4028GR-TR2
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 3253491)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================

一条出错的数据：
Messages: [{'content': "Let $a_1 < a_2 < \\cdots < a_k$ denote the sequence of all positive integers between $1$ and $91$ which are relatively prime to $91$, and set $\\omega = e^{2\\pi i/91}$. Define \\[S = \\prod_{1\\leq q < p\\leq k}(\\omega^{a_p} - \\omega^{a_q}).\\] Given that $S$ is a positive integer, compute the number of positive divisors of $S$.\n\nRemember to place the final answer in the last part using the format: \n\n\\boxed{{'The final answer goes here.'}}", 'role': 'user', 'tool_calls': None}
 {'content': "Okay, so I need to compute the number of positive divisors of the product S, which is defined as the product of all differences (ω^{a_p} - ω^{a_q}) where 1 ≤ q < p ≤ k. Here, ω is e^(2πi/91), and the a_i are the positive integers between 1 and 91 that are relatively prime to 91. First, let's make sure I understand the problem correctly.\n\nFirst, 91 is 7*13, so the numbers relatively prime to 91 between 1 and 91 are the numbers not divisible by 7 or 13. The Euler's totient function φ(91) = φ(7)φ(13) = 6*12 = 72. So there are 72 numbers between 1 and 91 that are coprime to 91. Therefore, k = 72, and the sequence a_1, a_2, ..., a_72 are these numbers in increasing order.\n\nThen, ω is a primitive 91st root of unity. The product S is over all 1 ≤ q < p ≤ k of (ω^{a_p} - ω^{a_q}). So this is the product of all differences between distinct roots of unity ω^{a_i} and ω^{a_j}, where a_i and a_j are coprime to 91.\n\nThis seems similar to the concept of the Vandermonde determinant, which is the product of (x_i - x_j) for i < j. The Vandermonde determinant is known for being the product of all differences, and its magnitude is the product of differences of the roots. However, in this case, the roots are not all roots of unity, but only those corresponding to exponents coprime to 91. So we are dealing with a subset of the 91st roots of unity, specifically the primitive roots? Wait, actually, ω is a primitive 91st root of unity, but ω^{a_i} where a_i is coprime to 91 would still be primitive 91st roots of unity. Because if a_i is coprime to 91, then the order of ω^{a_i} is 91. So the set {ω^{a_i}} is exactly the set of primitive 91st roots of unity. Therefore, the product S is the product of differences between all pairs of distinct primitive 91st roots of unity.\n\nHmm, so S is like the Vandermonde determinant for primitive 91st roots of unity. Now, I need to compute S and find the number of positive divisors of S, given that S is a positive integer.\n\nFirst, I need to figure out what S actually is. Since S is a product of complex numbers, but the problem states that S is a positive integer. Therefore, all the complex factors must multiply to a positive integer. But how?\n\nFirst, note that the product S is a product of terms of the form (ω^{a_p} - ω^{a_q}). Each term is a complex number. The product of all such terms is a complex number. But the problem says it's a positive integer. So perhaps there's some symmetry or property that causes all the imaginary parts to cancel out, and the product is real and positive. Also, since ω is on the unit circle, each term (ω^{a_p} - ω^{a_q}) has magnitude equal to the chord length between the two roots, which is 2*sin(π |a_p - a_q| /91). But the product of these magnitudes would be a real positive number. However, the actual product S is a product of complex numbers, which may have phases, but the problem states that S is a positive integer, so the product must end up being a real positive integer. Therefore, perhaps S is equal to the absolute value of the product, but that might not necessarily be an integer. Alternatively, maybe the product S is a real number, which is an integer, and positive.\n\nAlternatively, perhaps the product can be related to some algebraic integer, and since it's a positive real algebraic integer, it must be a rational integer. So maybe S is the square of the Vandermonde determinant over the primitive roots, or something like that?\n\nAlternatively, maybe S is related to the discriminant of a cyclotomic field. The discriminant of a number field is related to the product of differences of conjugates. Specifically, the discriminant of the cyclotomic field Q(ζ_n) is known, and it involves the product of differences of primitive roots of unity. So perhaps S is related to that discriminant. Let me recall.\n\nThe discriminant D of the cyclotomic field Q(ζ_n) is given by (-1)^φ(n)/2 * n^{φ(n)} / product_{p | n} p^{φ(n)/(p-1)} }.\n\nWait, but maybe more directly, the discriminant is equal to the square of the Vandermonde determinant formed by the primitive roots. Wait, actually, the discriminant of a polynomial is the product of squares of differences of roots, which is the square of the Vandermonde determinant. But here, S is the product of differences, not the square. So if the polynomial whose roots are the primitive roots of unity is the cyclotomic polynomial Φ_n(x), then the discriminant of Φ_n(x) is equal to the square of the Vandermonde determinant divided by the leading coefficient squared, but since Φ_n(x) is monic, it's just the square of the product of differences of roots. Wait, no, actually, the discriminant is the product_{i < j} (α_i - α_j)^2 where α_i are the roots. Therefore, the discriminant would be S^2, where S is the Vandermonde determinant. However, in our case, S is the product over all 1 ≤ q < p ≤ k (ω^{a_p} - ω^{a_q}), so S is the Vandermonde determinant. Therefore, the discriminant of the cyclotomic polynomial Φ_91(x) would be S^2. Therefore, S^2 is the discriminant of Φ_91(x), which is known. Therefore, maybe we can compute S^2 and then take the square root, but S is supposed to be a positive integer, so S would be the positive square root of the discriminant of Φ_91(x).\n\nTherefore, perhaps the value of S can be determined via the discriminant of the cyclotomic polynomial. Let me look up the formula for the discriminant of cyclotomic polynomials.\n\nFrom some references, the discriminant D of Φ_n(x) is given by:\n\nD = (-1)^φ(n)/2 * \\frac{n^{\\varphi(n)}}{\\prod_{p | n} p^{\\varphi(n)/(p-1)}}}.\n\nWait, let me check. For the cyclotomic polynomial Φ_n(x), the discriminant is known and there's a formula. According to some sources, the discriminant D of Φ_n(x) is:\n\nD = (-1)^{φ(n)/2} \\cdot \\frac{n^{\\varphi(n)}}{\\prod_{p | n} p^{\\varphi(n)/(p-1)}}}}\n\nYes, that seems to be the case. For example, for prime n, Φ_p(x) has discriminant (-1)^{(p-1)/2} p^{p-2}. So in that case, φ(p) = p-1, so exponent on n is φ(n), which is p-1, and the denominator is p^{φ(n)/(p-1)} = p^{(p-1)/(p-1)} = p^1. So D = (-1)^{(p-1)/2} p^{p-2}, which matches the known discriminant for prime cyclotomic polynomials.\n\nSo in general, for n = product of primes, the discriminant is (-1)^{φ(n)/2} times n^{φ(n)} divided by the product over primes p dividing n of p^{φ(n)/(p-1)}.\n\nIn our case, n = 91 = 7*13. Then φ(n) = φ(7)φ(13) = 6*12 = 72. So φ(n)/2 = 36. So the discriminant D would be (-1)^{36} * 91^{72} / (7^{72/(7-1)} * 13^{72/(13-1)}).\n\nCompute the exponents in the denominator:", 'role': 'assistant', 'tool_calls': array([{'function': {'arguments': {'code': 'n = 91\nphi_n = 72\nprimes = [7, 13]\nexponents_denominator = [phi_n // (p - 1) for p in primes]\nprint(exponents_denominator)'}, 'name': 'code_interpreter'}, 'type': 'function'}],
       dtype=object)}
 {'content': '[12, 6]', 'role': 'tool', 'tool_calls': None}
 {'content': "For p = 7, exponent is 72/(7-1) = 72/6 = 12.\n\nFor p = 13, exponent is 72/(13-1) = 72/12 = 6.\n\nTherefore, D = 1 * 91^{72} / (7^{12} * 13^{6}).\n\nSimplify 91^{72} = (7*13)^{72} = 7^{72} * 13^{72}.\n\nTherefore, D = (7^{72} * 13^{72}) / (7^{12} * 13^6) ) = 7^{60} * 13^{66}.\n\nBut D is equal to S^2. Therefore, S^2 = 7^{60} * 13^{66}, so S = 7^{30} * 13^{33}.\n\nBut wait, D is equal to the discriminant of Φ_n(x), which is equal to S^2, where S is the Vandermonde determinant, i.e., the product of (ω^{a_p} - ω^{a_q}) over 1 ≤ q < p ≤ k. However, in our problem, S is defined as that product, so S is the Vandermonde determinant. Therefore, S^2 is the discriminant D, which is 7^{60} * 13^{66}. Therefore, S is the square root of that, so S = 7^{30} * 13^{33}. Since the discriminant is positive (as (-1)^{36} is 1), so S is a positive real number. Therefore, S is 7^{30} * 13^{33}. But 7^{30} * 13^{33} is indeed an integer, so that's why S is a positive integer. So then, the problem is asking for the number of positive divisors of S.\n\nSo the number of positive divisors is given by multiplying one more than each exponent in the prime factorization. The prime factors of S are 7 and 13. The exponents are 30 and 33. Therefore, the number of positive divisors is (30 + 1)*(33 + 1) = 31*34.", 'role': 'assistant', 'tool_calls': array([{'function': {'arguments': {'code': 'exponents = [30, 33]\nnum_divisors = (exponents[0] + 1) * (exponents[1] + 1)\nprint(num_divisors)'}, 'name': 'code_interpreter'}, 'type': 'function'}],
       dtype=object)}
 {'content': '1054', 'role': 'tool', 'tool_calls': None}
 {'content': 'Compute that: 31*34 = 1054. Therefore, the answer is 1054.\n\nTherefore, the number of positive divisors is 1054. So the answer is 1054.\n\n\n\\boxed{1054}', 'role': 'assistant', 'tool_calls': None}]
Tools: [{"type": "function", "function": {"name": "code_interpreter", "description": "A tool for executing code.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to execute."}}, "required": ["code"]}}}]
Enable thinking: None

```

## 5. Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
禁用
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "你的模型路径",
    sliding_window=None
)
或者，代替默认 sdpa 后端，使用支持 sliding window 的实现
model = AutoModelForCausalLM.from_pretrained(
    "深度模型名称",
    attn_implementation="flash_attention_2",
    torch_dtype="torch.bfloat16",  # 或你模型指定的 dtype
    device_map="auto"
)


## 6. 并行GPU数量
RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
ulysses_sequence_parallel_size
Ulysses 的序列并行维度 ulysses_sequence_parallel_size 超出了实际 GPU 数量，构建 device mesh 会失败。
CUDA_VISIBLE_DEVICES=1,2  # 实际只有 2 块卡
torchrun --nproc_per_node=2
改成 ulysses_sequence_parallel_size=2

## 7. 容器内的代理配置，配置成主机的代理接口
ALL_PROXY=http://192.168.100.80:7890
HTTP_PROXY=http://192.168.100.8:7890
HTTPS_PROXY=http://192.168.100.8:7890
http_proxy=http://192.168.100.8:7890
https_proxy=http://192.168.100.8:7890


## 8. trainer nnodes报错，检查$ARNOLD_WORKER_NUM是否有值
ray.exceptions.RayTaskError(TypeError): ray::TaskRunner.run() (pid=51454, ip=192.168.100.8, actor_id=25af338a4901e3bf7de0eafb01000000, repr=<main_ppo.TaskRunner object at 0x73c820168850>)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 182, in run
    global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
TypeError: can't multiply sequence by non-int of type 'str'

trainer.nnodes=$ARNOLD_WORKER_NUM 


## 9. 报错,Reward函数没有找到
ray.exceptions.RayTaskError(FileNotFoundError): ray::TaskRunner.run() (pid=57583, ip=192.168.100.8, actor_id=21407495e89bd14ae1d0a83e01000000, repr=<main_ppo.TaskRunner object at 0x7f17c7364910>)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 211, in run
    reward_fn = load_reward_manager(
  File "/workspace/verl/verl/verl/trainer/ppo/reward.py", line 111, in load_reward_manager
    compute_score = get_custom_reward_fn(config)
  File "/workspace/verl/verl/verl/trainer/ppo/reward.py", line 62, in get_custom_reward_fn
    raise FileNotFoundError(f"Reward function file '{file_path}' not found.")
FileNotFoundError: Reward function file 'recipe/retool/retool.py' not found.

去掉retool.py前面的路径
data.custom_cls.path=retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=retool.py \


## 10.报错， 
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ValueError): ray::TaskRunner.run() (pid=63702, ip=192.168.100.8, actor_id=2255078b5577b0a9ae53a50f01000000, repr=<main_ppo.TaskRunner object at 0x7ad4abf50970>)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 222, in run
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 287, in create_rl_dataset
    dataset = dataset_cls(
  File "/workspace/verl/verl/verl/utils/dataset/rl_dataset.py", line 121, in __init__
    self._read_files_and_tokenize()
  File "/workspace/verl/backend/reTool/retool.py", line 71, in _read_files_and_tokenize
    dataframe = datasets.load_dataset(parquet_file)["train"]
  File "/usr/local/lib/python3.10/dist-packages/datasets/load.py", line 2084, in load_dataset
    builder_instance.download_and_prepare(
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 925, in download_and_prepare
    self._download_and_prepare(
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 979, in _download_and_prepare
    split_generators = self._split_generators(dl_manager, **split_generators_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/datasets/packaged_modules/parquet/parquet.py", line 60, in _split_generators
    self.info.features = datasets.Features.from_arrow_schema(pq.read_schema(f))
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 1850, in from_arrow_schema
    metadata_features = Features.from_dict(metadata["info"]["features"])
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 1888, in from_dict
    obj = generate_from_dict(dic)
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 1468, in generate_from_dict
    return {key: generate_from_dict(value) for key, value in obj.items()}
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 1468, in <dictcomp>
    return {key: generate_from_dict(value) for key, value in obj.items()}
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 1474, in generate_from_dict
    raise ValueError(f"Feature type '{_type}' not found. Available feature types: {list(_FEATURE_TYPES.keys())}")
ValueError: Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation', 'TranslationVariableLanguages', 'LargeList', 'Sequence', 'Array2D', 'Array3D', 'Array4D', 'Array5D', 'Audio', 'Image', 'Video', 'Pdf']
解决办法，使用相同的datasets
pip freeze | grep datasets
datasets==4.0.0

pip freeze | grep datasets
datasets==3.6.0

## 11.训练代码报错，检查下trainer.n_gpus_per_node的值是否不为空
Traceback (most recent call last):
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 40, in main
    run_ppo(config)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 77, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 2822, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 930, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(TypeError): ray::TaskRunner.run() (pid=109884, ip=192.168.100.8, actor_id=784a0f42f766c312b7850c1b01000000, repr=<main_ppo.TaskRunner object at 0x762381f4c910>)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 227, in run
    trainer = RayPPOTrainer(
  File "/workspace/verl/verl/verl/trainer/ppo/ray_trainer.py", line 384, in __init__
    self._validate_config()
  File "/workspace/verl/verl/verl/trainer/ppo/ray_trainer.py", line 411, in _validate_config
    assert real_train_batch_size % minimal_bsz == 0, (
TypeError: unsupported operand type(s) for %: 'int' and 'str'

## 12.训练报错， train_sp=2 模型并行需要和GPU数量一致
(WorkerDict pid=136980) Exception raised in creation task: The actor died because of an error raised in its creation task, ray::s6M6paWorkerDict_0:0:WorkerDict.__init__() (pid=136980, ip=192.168.100.8, actor_id=802ec5a7e3e04c21babf95d701000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x721457a7cc70>)
(WorkerDict pid=136980)   File "/usr/lib/python3.10/concurrent/futures/_base.py", line 451, in result
(WorkerDict pid=136980)     return self.__get_result()
(WorkerDict pid=136980)   File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
(WorkerDict pid=136980)     raise self._exception
(WorkerDict pid=136980)   File "/workspace/verl/verl/verl/single_controller/ray/base.py", line 799, in __init__
(WorkerDict pid=136980)     self.worker_dict[key] = user_defined_cls(
(WorkerDict pid=136980)   File "/workspace/verl/verl/verl/workers/fsdp_workers.py", line 140, in __init__
(WorkerDict pid=136980)     self.ulysses_device_mesh = init_device_mesh(
(WorkerDict pid=136980)   File "/usr/local/lib/python3.10/dist-packages/torch/distributed/device_mesh.py", line 1003, in init_device_mesh
(WorkerDict pid=136980)     device_mesh = DeviceMesh(
(WorkerDict pid=136980)   File "/usr/local/lib/python3.10/dist-packages/torch/distributed/device_mesh.py", line 453, in __init__
(WorkerDict pid=136980)     self._init_process_groups()
(WorkerDict pid=136980)   File "/usr/local/lib/python3.10/dist-packages/torch/distributed/device_mesh.py", line 530, in _init_process_groups
(WorkerDict pid=136980)     pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(
(WorkerDict pid=136980) RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
Traceback (most recent call last):
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 40, in main
    run_ppo(config)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 77, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 2822, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 930, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ActorDiedError): ray::TaskRunner.run() (pid=134546, ip=192.168.100.8, actor_id=b7edf6d4d6e698cba0aa6a5b01000000, repr=<main_ppo.TaskRunner object at 0x70b02c530970>)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 242, in run
    trainer.init_workers()
  File "/workspace/verl/verl/verl/trainer/ppo/ray_trainer.py", line 866, in init_workers
    self.actor_rollout_wg.init_model()
  File "/workspace/verl/verl/verl/single_controller/ray/base.py", line 50, in __call__
    output = ray.get(output)
ray.exceptions.ActorDiedError: The actor died because of an error raised in its creation task, ray::s6M6paWorkerDict_0:0:WorkerDict.__init__() (pid=136980, ip=192.168.100.8, actor_id=802ec5a7e3e04c21babf95d701000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x721457a7cc70>)
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 451, in result
    return self.__get_result()
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
  File "/workspace/verl/verl/verl/single_controller/ray/base.py", line 799, in __init__
    self.worker_dict[key] = user_defined_cls(
  File "/workspace/verl/verl/verl/workers/fsdp_workers.py", line 140, in __init__
    self.ulysses_device_mesh = init_device_mesh(
  File "/usr/local/lib/python3.10/dist-packages/torch/distributed/device_mesh.py", line 1003, in init_device_mesh
    device_mesh = DeviceMesh(
  File "/usr/local/lib/python3.10/dist-packages/torch/distributed/device_mesh.py", line 453, in __init__
    self._init_process_groups()
  File "/usr/local/lib/python3.10/dist-packages/torch/distributed/device_mesh.py", line 530, in _init_process_groups
    pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(
RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0] because the unspecified dimension size -1 can be any value and is ambiguous

## 13.报错
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 4260, in from_pretrained
    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 952, in _get_resolved_checkpoint_files
    raise EnvironmentError(
OSError: Error no file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt.index or flax_model.msgpack found in directory /workspace/verl/backend/reTool/checkpoint/multiturn-sft-Qwen2.5-0.5B-Instruct/global_step_250/huggingface.

## 14. PPO训练时报错，显存不足，指定显卡或者清理显存。 export CUDA_VISIBLE_DEVICES=1,2
Traceback (most recent call last):
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 40, in main
    run_ppo(config)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 77, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 2822, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 930, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ValueError): ray::TaskRunner.run() (pid=239082, ip=192.168.100.8, actor_id=694a55ad70568964cb986ed701000000, repr=<main_ppo.TaskRunner object at 0x7241c1b089d0>)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 242, in run
    trainer.init_workers()
  File "/workspace/verl/verl/verl/trainer/ppo/ray_trainer.py", line 866, in init_workers
    self.actor_rollout_wg.init_model()
  File "/workspace/verl/verl/verl/single_controller/ray/base.py", line 50, in __call__
    output = ray.get(output)
ray.exceptions.RayTaskError(ValueError): ray::WorkerDict.actor_rollout_init_model() (pid=239450, ip=192.168.100.8, actor_id=415e97a811c421b11a495d6b01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x792471eca680>)
  File "/workspace/verl/verl/verl/single_controller/ray/base.py", line 720, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/workspace/verl/verl/verl/single_controller/base/decorator.py", line 514, in inner
    return func(*args, **kwargs)
  File "/workspace/verl/verl/verl/workers/fsdp_workers.py", line 628, in init_model
    self.rollout, self.rollout_sharding_manager = self._build_rollout(
  File "/workspace/verl/verl/verl/workers/fsdp_workers.py", line 501, in _build_rollout
    rollout = vllm_rollout_cls(
  File "/workspace/verl/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py", line 166, in __init__
    self.inference_engine = LLM(
  File "/usr/local/lib/python3.10/dist-packages/vllm/utils.py", line 1161, in inner
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py", line 247, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 510, in from_engine_args
    return engine_cls.from_vllm_config(
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/llm_engine.py", line 112, in from_vllm_config
    return cls(vllm_config=vllm_config,
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/llm_engine.py", line 92, in __init__
    self.engine_core = EngineCoreClient.make_client(
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core_client.py", line 75, in make_client
    return InprocClient(vllm_config, executor_class, log_stats)
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core_client.py", line 198, in __init__
    self.engine_core = EngineCore(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core.py", line 71, in __init__
    self._initialize_kv_caches(vllm_config)
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core.py", line 133, in _initialize_kv_caches
    kv_cache_configs = [
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core.py", line 134, in <listcomp>
    get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/core/kv_cache_utils.py", line 699, in get_kv_cache_config
    check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory)
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/core/kv_cache_utils.py", line 527, in check_enough_kv_cache_memory
    raise ValueError("No available memory for the cache blocks. "
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.

# 15.  PPO最后训练阶段报错，这个问题不用管，已经训完了，关闭进程的时候有错误，但是没啥影响
0.011548247188329697 - critic/returns/max:2.4748666286468506 - critic/returns/min:-1.6201815605163574 - response_length/mean:87.4609375 - r
esponse_length/max:128.0 - response_length/min:39.0 - response_length/clip_ratio:0.2265625 - prompt_length/mean:57.875 - prompt_length/max:
73.0 - prompt_length/min:47.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.00016492418944835663 - timing_s/generate_sequences:
4.5007100105285645 - timing_s/reshard:0.33666151762008667 - timing_s/gen:5.458714358508587 - timing_s/reward:0.06054678000509739 - timing_s
/old_log_prob:0.5929685160517693 - timing_s/adv:0.012546919286251068 - timing_s/update_actor:2.6866441518068314 - timing_s/testing:21.20649
6983766556 - timing_s/step:30.030646292492747 - timing_s/stop_profile:9.175390005111694e-05 - timing_per_token_ms/gen:0.4876028904429287 -
timing_per_token_ms/update_actor:0.1444199404293303 - timing_per_token_ms/adv:0.0006744567696743035 - perf/total_num_tokens:18603 - perf/ti
me_per_step:30.030646292492747 - perf/throughput:619.4671875793262
(TaskRunner pid=267532) ("Final validation metrics: {'val-aux/char_count/reward/mean@1': "
(TaskRunner pid=267532)  "0.2008238928939238, 'val-aux/char_count/reward/mean@2': 0.0, "
(TaskRunner pid=267532)  "'val-aux/char_count/reward/std@2': 0.0, "
(TaskRunner pid=267532)  "'val-aux/char_count/reward/best@2/mean': 0.0, "
(TaskRunner pid=267532)  "'val-aux/char_count/reward/best@2/std': 0.0, "
(TaskRunner pid=267532)  "'val-aux/char_count/reward/worst@2/mean': 0.0, "
(TaskRunner pid=267532)  "'val-aux/char_count/reward/worst@2/std': 0.0, "
(TaskRunner pid=267532)  "'val-core/char_count/reward/mean@3': 0.0, 'val-aux/char_count/reward/std@3': "
(TaskRunner pid=267532)  "0.0, 'val-core/char_count/reward/best@3/mean': 0.0, "
(TaskRunner pid=267532)  "'val-core/char_count/reward/best@3/std': 0.0, "
(TaskRunner pid=267532)  "'val-aux/char_count/reward/worst@3/mean': 0.0, "
(TaskRunner pid=267532)  "'val-aux/char_count/reward/worst@3/std': 0.0}")
(raylet) A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffffc90bbe1574830f10a33b147c01000000 Worker ID: f7ea96b1e8e804f578390a97c97d6bf28b5bc9fed13f0ef433e444c7 Node ID: cc69c4cc97291a2847bc91bcfbd1f19c419b6e6fbe86aad345f2f665 Worker IP address: 192.168.100.8 Worker port: 41497 Worker PID: 269812 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly by a signal. SystemExit is raised (sys.exit is called). Exit code: 1. The process receives a SIGTERM.

# 16. PPO训练后的模型没有保存，修改save_freq为1，默认保存到当前的checkpoints/verl_example/smol135m_grpo目录下
trainer.save_freq=1

# 17 Ray报错， 修改infer_tp=1, 降低 infer_tp 以匹配实际 GPU 数量,如果你当前只打算使用 2 张 GPU，直接将 infer_tp 改成 1 或 2：,infer_tp 代表 tensor parallel size，是用于推理的模型切分。
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'algorithm.use_kl_in_reward=False', 'algorithm.kl_ctrl.kl_coef=0.0', "data.train_files=['/workspace/verl/backend/reTool/dataset/BytedTsinghua/train']", "data.val_files=['/workspace/verl/backend/reTool/dataset/Maxwell/validation']", 'data.return_raw_chat=True', 'data.train_batch_size=16', 'data.max_prompt_length=2048', 'data.max_response_length=16384', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'data.custom_cls.path=retool.py', 'data.custom_cls.name=CustomRLHFDataset', 'custom_reward_function.path=retool.py', 'custom_reward_function.name=compute_score', 'actor_rollout_ref.model.path=./models/merged_sft_model', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.use_kl_loss=False', 'actor_rollout_ref.actor.kl_loss_coef=0.0', 'actor_rollout_ref.actor.clip_ratio_low=0.2', 'actor_rollout_ref.actor.clip_ratio_high=0.28', 'actor_rollout_ref.actor.clip_ratio_c=10.0', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.use_dynamic_bsz=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=8', 'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432', 'actor_rollout_ref.actor.ulysses_sequence_parallel_size=2', 'actor_rollout_ref.actor.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=True', 'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=73728', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.mode=async', 'actor_rollout_ref.rollout.tensor_model_parallel_size=4', 'actor_rollout_ref.rollout.multi_turn.enable=True', 'actor_rollout_ref.rollout.multi_turn.max_user_turns=8', 'actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8', 'actor_rollout_ref.rollout.multi_turn.tool_config_path=./sandbox_fusion_tool_config.yaml', 'actor_rollout_ref.rollout.multi_turn.format=hermes', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.9', 'actor_rollout_ref.rollout.n=16', 'actor_rollout_ref.rollout.val_kwargs.top_p=0.6', 'actor_rollout_ref.rollout.val_kwargs.temperature=1.0', 'actor_rollout_ref.rollout.val_kwargs.n=30', 'trainer.logger=[console]', 'trainer.project_name=wuxibin_retool', 'trainer.experiment_name=qwen2.5-05b_dapo', 'trainer.n_gpus_per_node=2', 'trainer.val_before_train=True', 'trainer.log_val_generations=100', 'trainer.nnodes=1', 'trainer.save_freq=1', 'trainer.default_local_dir=/workspace/verl/backend/reTool/checkpoint/qwen2.5-05b_dapo', 'trainer.test_freq=5', 'trainer.total_epochs=1']
Traceback (most recent call last):
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 40, in main
    run_ppo(config)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 77, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 2822, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 930, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(AssertionError): ray::TaskRunner.run() (pid=576436, ip=192.168.100.8, actor_id=90aa9fb96c4861d1b90e76ee01000000, repr=<main_ppo.TaskRunner object at 0x72c1a8134940>)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 242, in run
    trainer.init_workers()
  File "/workspace/verl/verl/verl/trainer/ppo/ray_trainer.py", line 866, in init_workers
    self.actor_rollout_wg.init_model()
  File "/workspace/verl/verl/verl/single_controller/ray/base.py", line 50, in __call__
    output = ray.get(output)
ray.exceptions.RayTaskError(AssertionError): ray::WorkerDict.actor_rollout_init_model() (pid=578871, ip=192.168.100.8, actor_id=6af2d7d16b4888d85e0ac3fa01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7a0ebad44c10>)
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 451, in result
    return self.__get_result()
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
  File "/workspace/verl/verl/verl/single_controller/ray/base.py", line 720, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/workspace/verl/verl/verl/single_controller/base/decorator.py", line 514, in inner
    return func(*args, **kwargs)
  File "/workspace/verl/verl/verl/workers/fsdp_workers.py", line 628, in init_model
    self.rollout, self.rollout_sharding_manager = self._build_rollout(
  File "/workspace/verl/verl/verl/workers/fsdp_workers.py", line 1654, in _build_rollout
    rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)
  File "/workspace/verl/verl/verl/workers/fsdp_workers.py", line 471, in _build_rollout
    assert self.world_size % infer_tp == 0, (
AssertionError: rollout world_size: 2 is not divisible by infer_tp: 4

# 16.  export VLLM_USE_V1=1, 调用了 AsyncLLM.from_vllm_config()，它默认使用 V1 引擎逻辑。但当前环境变量 VLLM_USE_V1 被设置为 False（也可能是没设，默认就是 False），导致逻辑冲突，vLLM 库抛出了异常。
]
Traceback (most recent call last):
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 40, in main
    run_ppo(config)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 77, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 2822, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/usr/local/lib/python3.10/dist-packages/ray/_private/worker.py", line 930, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ValueError): ray::TaskRunner.run() (pid=585134, ip=192.168.100.8, actor_id=2b9e243534fd3aaadd6bf62801000000, repr=<main_ppo.TaskRunner object at 0x727cd793c9a0>)
  File "/workspace/verl/verl/verl/trainer/main_ppo.py", line 242, in run
    trainer.init_workers()
  File "/workspace/verl/verl/verl/trainer/ppo/ray_trainer.py", line 874, in init_workers
    self.async_rollout_manager = AgentLoopManager(
  File "/workspace/verl/verl/verl/experimental/agent_loop/agent_loop.py", line 461, in __init__
    self._initialize_llm_servers()
  File "/workspace/verl/verl/verl/experimental/agent_loop/agent_loop.py", line 513, in _initialize_llm_servers
    ray.get([server.init_engine.remote() for server in self.async_llm_servers])
ray.exceptions.RayTaskError(ValueError): ray::AsyncvLLMServer.init_engine() (pid=588112, ip=192.168.100.8, actor_id=adae42d67fca8af1305046a501000000, repr=<verl.workers.rollout.vllm_rollout.vllm_async_server.AsyncvLLMServer object at 0x71ab37a45f30>)
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 451, in result
    return self.__get_result()
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
  File "/workspace/verl/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py", line 267, in init_engine
    self.engine = AsyncLLM.from_vllm_config(vllm_config)
  File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/async_llm.py", line 143, in from_vllm_config
    raise ValueError(
ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. This should not happen. As a workaround, try using AsyncLLMEngine.from_vllm_config(...) or explicitly set VLLM_USE_V1=0 or 1 and report this issue on Github.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
(TaskRunner pid=585134) Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): ray::AsyncvLLMServer.init_engine() (pid=588113, ip=192.168.100.8, actor_id=25b097a4fc28b46f978580db01000000, repr=<verl.workers.rollout.vllm_rollout.vllm_async_server.AsyncvLLMServer object at 0x72475454df30>)
(TaskRunner pid=585134)   File "/usr/lib/python3.10/concurrent/futures/_base.py", line 451, in result
(TaskRunner pid=585134)     return self.__get_result()
(TaskRunner pid=585134)   File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
(TaskRunner pid=585134)     raise self._exception
(TaskRunner pid=585134)   File "/workspace/verl/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py", line 267, in init_engine
(TaskRunner pid=585134)     self.engine = AsyncLLM.from_vllm_config(vllm_config)
(TaskRunner pid=585134)   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/async_llm.py", line 143, in from_vllm_config
(TaskRunner pid=585134)     raise ValueError(
(TaskRunner pid=585134) ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. This should not happen. As a workaround, try using AsyncLLMEngine.from_vllm_config(...) or explicitly set VLLM_USE_V1=0 or 1 and report this issue on Github.
(AsyncvLLMServer pid=588113) WARNING 08-05 01:28:55 [cuda.py:93] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
(AsyncvLLMServer pid=588113) instance_id: 6b1f20db-4134-41d9-a084-253ff6bd41a1:MT87lQ:2:0 initializes with external actors: ['MT87lQWorkerDict_0:0']
(AsyncvLLMServer pid=588113) VERL_VLLM_ZMQ_ADDRESSES: ['ipc:///tmp/verl_vllm_zmq_587568_root.ipc']
(AsyncvLLMServer pid=588113) Using blocking ray.get inside async actor. This blocks the event loop. Please use `await` on object ref with asyncio.gather if you want to yield execution to the event loop instead.