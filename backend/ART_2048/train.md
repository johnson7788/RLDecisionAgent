python train.py
Registering model...，进行super中..
调用backend的_prepare_backend_for_training
INFO 08-20 21:32:03 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 08-20 21:32:03 [__init__.py:239] Automatically detected platform cuda.
⚙️  Running in WANDB offline mode
_prepare_backend_for_training启动OpenAI服务: None
DEBUG put request: start_openai_server () {'config': None}
/usr/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 8 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
/workspace/verl/ART/src/art/__init__.py:10: UserWarning: WARNING: Unsloth should be imported before transformers, peft to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  import unsloth  # type: ignore # noqa: F401
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
⚙️  Running in WANDB offline mode
INFO 08-20 21:32:15 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 08-20 21:32:15 [__init__.py:239] Automatically detected platform cuda.
🦥 Unsloth Zoo will now patch everything to make training faster!
[HANDLE_REQUEST] id=9d6695c1-0744-43cd-9810-0b3b2f931a26, method=start_openai_server, args=(), kwargs={'config': None}
[CALLABLE] Executing start_openai_server
[COROUTINE] Awaiting start_openai_server
[start_openai_server] 开始启动 OpenAI server...
[start_openai_server] 当前 output_dir: /workspace/verl/ART/.art/2048/models/tutorial-001
[start_openai_server] 传入 config: None
[start_openai_server] get_last_checkpoint_dir 返回: None
[start_openai_server] 未找到 checkpoint，尝试使用 step=0 checkpoint
[start_openai_server] step=0 checkpoint 路径: /workspace/verl/ART/.art/2048/models/tutorial-001/checkpoints/0000
[start_openai_server] 已确保目录存在: /workspace/verl/ART/.art/2048/models/tutorial-001/checkpoints
[start_openai_server] 保存初始模型中...
Unsloth: Patching vLLM v1 graph capture
Unsloth: Patching vLLM v0 graph capture
==((====))==  Unsloth 2025.8.6: Fast Qwen2 patching. Transformers: 4.55.2. vLLM: 0.8.5.post1.
   \\   /|    NVIDIA GeForce RTX 4090 D. Num GPUs = 3. Max memory: 23.546 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.6.0+cu124. CUDA: 8.9. CUDA Toolkit: 12.4. Triton: 3.2.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post2. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: vLLM loading unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit with actual GPU utilization = 22.37%
Unsloth: Your GPU has CUDA compute capability 8.9 with VRAM = 23.55 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 8192. Num Sequences = 192.
Unsloth: vLLM's KV Cache can use up to 4.8 GB. Also swap space = 6 GB.
INFO 08-20 21:32:53 [config.py:717] This model supports multiple tasks: {'reward', 'generate', 'embed', 'score', 'classify'}. Defaulting to 'generate'.
Unsloth: vLLM Bitsandbytes config using kwargs = {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_quant_storage': 'uint8', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': ['lm_head', 'multi_modal_projector', 'merger', 'modality_projection', 'model.layers.0.self_attn', 'model.layers.0.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.21.mlp', 'model.layers.0.self_attn.q_proj'], 'llm_int8_threshold': 6.0}
INFO 08-20 21:32:53 [llm_engine.py:240] Initializing a V0 LLM engine (v0.8.5.post1) with config: model='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', speculative_config=None, tokenizer='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=bitsandbytes, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='auto', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit, num_scheduler_steps=16, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"level":0,"backend":"inductor","splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"epilogue_fusion":true,"max_autotune":false,"shape_padding":true,"trace.enabled":false,"triton.cudagraphs":true,"debug":false,"dce":true,"memory_planning":true,"coordinate_descent_tuning":true,"trace.graph_diagram":false,"compile_threads":32,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.multi_kernel":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,"triton.cooperative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchmark_combo_kernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":192}, use_cached_outputs=False,
INFO 08-20 21:32:59 [cuda.py:292] Using Flash Attention backend.
INFO 08-20 21:32:59 [cuda.py:292] Using Flash Attention backend.
INFO 08-20 21:33:10 [parallel_state.py:1004] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0
INFO 08-20 21:33:10 [model_runner.py:1108] Starting to load model unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit...
INFO 08-20 21:33:10 [loader.py:1187] Loading weights with BitsAndBytes quantization. May take a while ...
INFO 08-20 21:33:13 [weight_utils.py:265] Using model weights format ['*.safetensors']
INFO 08-20 21:33:14 [weight_utils.py:281] Time spent downloading weights for unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit: 0.557758 seconds
INFO 08-20 21:33:14 [weight_utils.py:315] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.07it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.07it/s]

Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.88it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.87it/s]

INFO 08-20 21:33:15 [punica_selector.py:18] Using PunicaWrapperGPU.
INFO 08-20 21:33:15 [model_runner.py:1140] Model loading took 0.5138 GiB and 5.444948 seconds
INFO 08-20 21:33:18 [worker.py:287] Memory profiling takes 2.38 seconds
INFO 08-20 21:33:18 [worker.py:287] the current vLLM instance can use total_gpu_memory (23.55GiB) x gpu_memory_utilization (0.22) = 5.27GiB
INFO 08-20 21:33:18 [worker.py:287] model weights take 0.51GiB; non_torch_memory takes 0.07GiB; PyTorch activation peak memory takes 1.06GiB; the rest of the memory reserved for KV Cache is 3.63GiB.
INFO 08-20 21:33:18 [executor_base.py:112] # cuda blocks: 19813, # CPU blocks: 32768
INFO 08-20 21:33:18 [executor_base.py:117] Maximum concurrency for 8192 tokens per request: 38.70x
INFO 08-20 21:33:22 [vllm_utils.py:671] Unsloth: Running patched vLLM v0 `capture_model`.
INFO 08-20 21:33:22 [model_runner.py:1450] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|██████████████████████████████████████████████████████████████████████████████| 27/27 [00:09<00:00,  2.93it/s]
INFO 08-20 21:33:31 [model_runner.py:1592] Graph capturing finished in 9 secs, took 0.40 GiB
INFO 08-20 21:33:31 [vllm_utils.py:678] Unsloth: Patched vLLM v0 graph capture finished in 9 secs.
INFO 08-20 21:33:32 [llm_engine.py:437] init engine (profile, create kv cache, warmup model) took 16.84 seconds
Unsloth: Just some info: will skip parsing ['k_norm', 'post_feedforward_layernorm', 'pre_feedforward_layernorm', 'q_norm']
Unsloth: Just some info: will skip parsing ['k_norm', 'post_feedforward_layernorm', 'pre_feedforward_layernorm', 'q_norm']
Unsloth 2025.8.6 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers.
[start_openai_server] 初始模型保存完成
[start_openai_server] 停止可能已有的 OpenAI server...
[start_openai_server] 旧的 OpenAI server 已停止
[start_openai_server] 准备启动新的 openai_server_task，配置如下：
  - model_name: tutorial-001
  - base_model: Qwen/Qwen2.5-0.5B-Instruct
  - log_file: /workspace/verl/ART/.art/2048/models/tutorial-001/logs/vllm.log
  - lora_path: /workspace/verl/ART/.art/2048/models/tutorial-001/checkpoints/0000
  - config: {'log_file': '/workspace/verl/ART/.art/2048/models/tutorial-001/logs/vllm.log', 'server_args': {'api_key': 'default', 'lora_modules': ['{"name": "tutorial-001", "path": "/workspace/verl/ART/.art/2048/models/tutorial-001/checkpoints/0000"}'], 'return_tokens_as_token_ids': True, 'enable_auto_tool_choice': True, 'tool_call_parser': 'hermes'}, 'engine_args': {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'num_scheduler_steps': 16, 'served_model_name': 'Qwen/Qwen2.5-0.5B-Instruct', 'disable_log_requests': True, 'generation_config': 'vllm'}}
[openai_server_task] 初始化开始...
[openai_server_task] 调用 subclass_chat_completion_request()...
[openai_server_task] 导入 vllm.entrypoints.openai.api_server...
[openai_server_task] 应用 vLLM patches...
[openai_server_task] 设置 vLLM 日志文件: /workspace/verl/ART/.art/2048/models/tutorial-001/logs/vllm.log
[openai_server_task] 启动 OpenAI 兼容 server 协程...
[openai_server_task] 使用 base_url=http://0.0.0.0:8000/v1, api_key=default
[openai_server_task] 等待 server 启动，超时时间: 30.0 秒
[openai_server_task] build_async_engine_client 被调用
[openai_server_task] add_lora 被调用
[LoRARequest] 获取属性: lora_int_id
[LoRARequest] 获取属性: lora_int_id
[LoRARequest] 获取属性: lora_path
[LoRARequest] 获取属性: lora_path
[LoRARequest] lora_tensors 不存在，返回 None
[LoRARequest] 获取属性: lora_int_id
[LoRARequest] 获取属性: lora_int_id
[openai_server_task] add_lora 完成
[openai_server_task.test_client] 开始轮询检查 server 是否可用...
[2025-08-20 21:33:58] INFO _client.py:1740: HTTP Request: GET http://0.0.0.0:8000/v1/models "HTTP/1.1 200 OK"
[openai_server_task.test_client] 成功连通 server！
[openai_server_task] task <Task finished name='Task-4' coro=<openai_server_task.<locals>.test_client() done, defined at /workspace/verl/ART/src/art/vllm/server.py:98> result=None> 已完成，检查结果...
[openai_server_task] server 成功启动！
[start_openai_server] openai_server_task 启动完成: <Task pending name='Task-3' coro=<run_server() running at /usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/api_server.py:1094> wait_for=<Task pending name='Task-6' coro=<Server.serve() running at /usr/local/lib/python3.10/dist-packages/uvicorn/server.py:70> wait_for=<Future pending cb=[Task.__wakeup()]> cb=[Task.__wakeup()]>>
[start_openai_server] 设置 lora: /workspace/verl/ART/.art/2048/models/tutorial-001/checkpoints/0000
[start_openai_server] 启动流程完成 ✅
[PICKLE] Result of start_openai_server is picklable (<class 'NoneType'>)
[HANDLE_REQUEST_DONE] id=9d6695c1-0744-43cd-9810-0b3b2f931a26, result_type=<class 'NoneType'>, exception_type=<class 'NoneType'>
[gather_trajectory_groups] Received 1 groups to process.
[gather_trajectory_groups] Wrapping groups into futures...
[gather_trajectory_groups] Total trajectories to gather: 18
gather:   0%|                                                                                                            | 0/18 [00:00<?, ?it/s][gather_trajectory_groups] Awaiting all futures...
[wrap_group_awaitable] Awaiting group
[DEBUG] Gather loop started, total futures: 18
[DEBUG] Waiting for future #0 ...
[2025-08-20 21:33:58] WARNING util.py:36: Warning: Traces will not be logged. Call weave.init to log your traces to a project.
 (subsequent messages of this type will be suppressed)
[RULER-DEBUG] Serialized trajectory 17: <trajectory id="17">
[{"content": "_|_|_|_\n2|_|_|_\n_|2|_|_\n_|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|_|_\n_|_|_|2\n_|_|_|2\n4|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|_|_\n_|_|2|2\n_|_|_|2\n_|_|_|4\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"_|_|_|_\n4|_|2|_\n2|_|_|_\n4|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|_|_\n_|_|4|2\n_|_|2|2\n_|_|_|4\n", "role": "user"}, {"role": "assistant", "content": "<move>down</move>"}, {"content":
"_|_|_|2\n_|_|_|_\n_|_|4|4\n_|_|2|4\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"2|_|_|4\n_|_|_|_\n8|_|_|_\n2|4|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|4\n_|_|_|_\n_|_|_|8\n_|2|2|4\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|2|2|4\n_|_|_|_\n_|_|_|8\n_|_|4|4\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|4|4\n2|_|_|_\n_|_|_|8\n_|_|_|8\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"2|_|_|8\n_|_|_|2\n_|_|_|8\n_|_|_|8\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"2|8|_|_\n2|_|_|2\n8|_|_|_\n8|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|8\n2|_|_|4\n_|_|_|8\n_|_|_|8\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"2|8|_|_\n2|4|2|_\n8|_|_|_\n8|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|8\n_|2|4|2\n_|_|_|8\n_|_|2|8\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"2|8|_|_\n2|4|2|4\n8|_|_|_\n2|8|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|8\n2|4|2|4\n_|_|_|8\n_|2|2|8\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"2|8|_|_\n2|4|2|4\n8|_|_|_\n4|8|_|2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|8\n2|4|2|4\n_|2|_|8\n_|4|8|2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|8\n2|4|2|4\n2|_|2|8\n_|4|8|2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|8\n2|4|2|4\n2|_|4|8\n_|4|8|2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|8\n2|4|2|4\n2|2|4|8\n_|4|8|2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|2|8\n2|4|2|4\n2|4|4|8\n_|4|8|2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|2|2|8\n2|4|2|4\n_|2|8|8\n_|4|8|2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| _| 4| 8\n 2|
4| 2| 4\n 2| _| 2|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| _| 4| 8\n 2| 4|
2| 4\n _| _| 4|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| 2| 4| 8\n 2| 4| 2|
4\n _| 2| 4|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| 2| 4| 8\n 2| 4| 2| 4\n
2| 2| 4|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| 2| 4| 8\n 2| 4| 2| 4\n 2|
4| 4|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 2| 4| 8\n 2| 4| 2| 4\n _| 2|
8|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| 4| 4| 8\n 2| 4| 2| 4\n 2| 2|
8|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| 2| 8| 8\n 2| 4| 2| 4\n _| 4|
8|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| _| 2|16\n 2| 4| 2| 4\n 2| 4|
8|16\n _| 4| 8| 2\n", "role": "user"}, {"role": "assistant", "content": "Note: This does not simulate 2048 in this example. It is provided for
reference."}]
</trajectory>
[RULER-DEBUG] Serialized trajectory 18: <trajectory id="18">
[{"content": "_|_|_|2\n_|_|_|_\n_|2|_|_\n_|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"2|_|_|_\n_|_|_|2\n2|_|_|_\n_|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|_|2\n_|_|_|2\n_|_|2|2\n_|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>down</move>"}, {"content":
"_|_|_|_\n_|_|_|_\n_|_|_|2\n_|2|2|4\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"_|_|2|_\n_|_|_|_\n2|_|_|_\n4|4|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|_|2\n_|_|4|_\n_|_|_|2\n_|_|_|8\n", "role": "user"}, {"role": "assistant", "content": "<move>up</move>"}, {"content":
"_|_|4|4\n_|_|_|8\n_|_|_|_\n4|_|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"8|_|_|_\n8|_|_|_\n_|_|_|_\n4|2|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|_|8\n_|4|_|8\n_|_|_|_\n_|_|4|2\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"8|_|_|_\n4|8|_|_\n2|_|_|_\n4|2|_|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|_|8\n_|_|4|8\n_|_|_|2\n2|_|4|2\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content":
"8|_|_|_\n4|8|_|_\n2|_|_|2\n2|4|2|_\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content":
"_|_|_|8\n_|_|4|8\n2|_|_|4\n_|2|4|2\n", "role": "user"}, {"role": "assistant", "content": "<move>up</move>"}, {"content": " 2| 2| 8|16\n 2| _|
_| 4\n _| _| _| 2\n _| _| _| _\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| 4| 8|16\n 2| _| 2|
4\n _| _| _| 2\n _| _| _| _\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content": " 4| 8|16| _\n 4| 4| _| _\n
2| _| _| _\n 4| _| _| _\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| 4| 8|16\n _| _| _| 8\n _|
2| _| 2\n _| _| _| 4\n", "role": "user"}, {"role": "assistant", "content": "<move>up</move>"}, {"content": " _| 4| 8|16\n _| 2| _| 8\n _| _| _|
2\n 2| _| _| 4\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| 4| 8|16\n _| _| 2| 8\n _| _| _| 2\n
2| _| 2| 4\n", "role": "user"}, {"role": "assistant", "content": "<move>down</move>"}, {"content": " _| _| _|16\n 2| _| _| 8\n _| _| 8| 2\n 2|
4| 4| 4\n", "role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content": "16| _| _| _\n 2| 8| _| _\n 8| 2| 4| _\n 2| 8|
4| _\n", "role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| _| _|16\n _| _| 2| 8\n _| 8| 2| 4\n _| 2| 8|
4\n", "role": "user"}, {"role": "assistant", "content": "<move>up</move>"}, {"content": " 2| 8| 4|16\n _| 2| 8| 8\n 2| _| _| 8\n _| _| _| _\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 8| 4|16\n _| _| 2|16\n _| 2| 2| 8\n _| _| _| _\n",
"role": "user"}, {"role": "assistant", "content": "<move>down</move>"}, {"content": " _| 4| _| _\n _| _| _| _\n _| 8| 4|32\n 2| 2| 4| 8\n",
"role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content": " 4| _| _| _\n _| 2| _| _\n 8| 4|32| _\n 4| 4| 8| _\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| _| _| 4\n _| _| _| 2\n 2| 8| 4|32\n _| _| 8| 8\n",
"role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content": " 4| _| _| _\n 2| 2| _| _\n 2| 8| 4|32\n16| _| _| _\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| _| 2| 4\n _| _| _| 4\n 2| 8| 4|32\n _| _| _|16\n",
"role": "user"}, {"role": "assistant", "content": "<move>down</move>"}, {"content": " 2| _| _| _\n _| _| _| 8\n _| _| 2|32\n 2| 8| 4|16\n",
"role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content": " 2| _| _| _\n 8| _| _| 2\n 2|32| _| _\n 2| 8| 4|16\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " _| _| _| 2\n _| 2| 8| 2\n _| _| 2|32\n 2| 8| 4|16\n",
"role": "user"}, {"role": "assistant", "content": "<move>up</move>"}, {"content": " 2| 2| 8| 4\n _| 8| 2|32\n _| _| 4|16\n _| _| 2| _\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 4| 8| 4\n _| 8| 2|32\n _| _| 4|16\n _| _| _| 2\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 4| 8| 4\n _| 8| 2|32\n 2| _| 4|16\n _| _| _| 2\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 4| 8| 4\n _| 8| 2|32\n 2| 2| 4|16\n _| _| _| 2\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 4| 8| 4\n _| 8| 2|32\n 2| 4| 4|16\n _| _| _| 2\n",
"role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content": " 2| 4| 8| 4\n 8| 2|32| _\n 2| 8|16| 4\n 2| _| _| _\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 4| 8| 4\n 2| 8| 2|32\n 2| 8|16| 4\n _| _| _| 2\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 4| 8| 4\n 2| 8| 2|32\n 2| 8|16| 4\n _| _| 2| 2\n",
"role": "user"}, {"role": "assistant", "content": "<move>right</move>"}, {"content": " 2| 4| 8| 4\n 2| 8| 2|32\n 2| 8|16| 4\n _| _| 2| 4\n",
"role": "user"}, {"role": "assistant", "content": "<move>left</move>"}, {"content": " 2| 4| 8| 4\n 2| 8| 2|32\n 2| 8|16| 4\n 2| 4| _| 2\n",
"role": "user"}, {"role": "assistant", "content": "<move>left</move>"}]
</trajectory>
[RULER-DEBUG] Judge prompt constructed.
[RULER-DEBUG] Sending 2 messages to model `openai/o4-mini`.
21:25:16 - LiteLLM:INFO: utils.py:2958 -
LiteLLM completion() model= o4-mini; provider = openai
[2025-08-20 21:25:16] INFO utils.py:2958:
LiteLLM completion() model= o4-mini; provider = openai
[2025-08-20 21:25:16] INFO _client.py:1740: HTTP Request: POST http://localhost:6688/chat/completions "HTTP/1.1 200 OK"
21:25:46 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: openai/o4-mini-2025-04-16
[2025-08-20 21:25:46] INFO cost_calculator.py:655: selected model name for cost calculation: openai/o4-mini-2025-04-16
[RULER-DEBUG] Got response with 1 choices.
[RULER-DEBUG] Raw model content: {"scores":[{"trajectory_id":"1","explanation":"Chose a reasonable move (left) but used the wrong XML wrapper
tag.","score":0.6},{"trajectory_id":"2","explanation":"Did not issue a valid move in the required XML
format.","score":0.0},{"trajectory_id":"3","explanation":"Output is nonsensical and not a valid
move.","score":0.0},{"trajectory_id":"4","explanation":"Made one valid <move>right</move> then failed to follow up with a proper
move.","score":0.2},{"trajectory_id":"5","explanation":"No valid move in the required XML
format.","score":0.0},{"trajectory_id":"6","explanation":"Recommended right but added extra text and did not stick to <move>
XML.","score":0.3},{"trajectory_id":"7","explanation":"Offered a valid first move but then asked for more information instead of giving next
move.","score":0.2},{"trajectory_id":"8","explanation":"Used an invalid direction (<move>east</move>), not one of the four allowed
moves.","score":0.0},{"trajectory_id":"9","explanation":"Never provided a correctly formatted move at the
end.","score":0.0},{"trajectory_id":"10","explanation":"Failed to supply a valid XML
move.","score":0.0},{"trajectory_id":"11","explanation":"Ended without a proper <move> XML
response.","score":0.0},{"trajectory_id":"12","explanation":"Stopped saying “No move” without providing a valid move
XML.","score":0.0},{"trajectory_id":"13","explanation":"Did not continue the game or give a final <move>
XML.","score":0.0},{"trajectory_id":"14","explanation":"Concluded with a message instead of a valid move
tag.","score":0.0},{"trajectory_id":"15","explanation":"Successfully played through to 2048 and gave the correct final <move>
XML.","score":1.0},{"trajectory_id":"16","explanation":"Provided many moves but ended with unrelated commentary instead of a proper
move.","score":0.1},{"trajectory_id":"17","explanation":"Consistently used valid <move> tags and ended with a proper move, though didn’t reach
2048.","score":0.8},{"trajectory_id":"18","explanation":"Played a coherent sequence with valid <move> tags, ending correctly despite not
achieving 2048.","score":0.8}]}
[RULER-DEBUG] Pretty-printed JSON:
{
    'scores': [
        {'trajectory_id': '1', 'explanation': 'Chose a reasonable move (left) but used the wrong XML wrapper tag.', 'score': 0.6},
        {'trajectory_id': '2', 'explanation': 'Did not issue a valid move in the required XML format.', 'score': 0.0},
        {'trajectory_id': '3', 'explanation': 'Output is nonsensical and not a valid move.', 'score': 0.0},
        {'trajectory_id': '4', 'explanation': 'Made one valid <move>right</move> then failed to follow up with a proper move.', 'score': 0.2},
        {'trajectory_id': '5', 'explanation': 'No valid move in the required XML format.', 'score': 0.0},
        {'trajectory_id': '6', 'explanation': 'Recommended right but added extra text and did not stick to <move> XML.', 'score': 0.3},
        {
            'trajectory_id': '7',
            'explanation': 'Offered a valid first move but then asked for more information instead of giving next move.',
            'score': 0.2
        },
        {
            'trajectory_id': '8',
            'explanation': 'Used an invalid direction (<move>east</move>), not one of the four allowed moves.',
            'score': 0.0
        },
        {'trajectory_id': '9', 'explanation': 'Never provided a correctly formatted move at the end.', 'score': 0.0},
        {'trajectory_id': '10', 'explanation': 'Failed to supply a valid XML move.', 'score': 0.0},
        {'trajectory_id': '11', 'explanation': 'Ended without a proper <move> XML response.', 'score': 0.0},
        {'trajectory_id': '12', 'explanation': 'Stopped saying “No move” without providing a valid move XML.', 'score': 0.0},
        {'trajectory_id': '13', 'explanation': 'Did not continue the game or give a final <move> XML.', 'score': 0.0},
        {'trajectory_id': '14', 'explanation': 'Concluded with a message instead of a valid move tag.', 'score': 0.0},
        {'trajectory_id': '15', 'explanation': 'Successfully played through to 2048 and gave the correct final <move> XML.', 'score': 1.0},
        {
            'trajectory_id': '16',
            'explanation': 'Provided many moves but ended with unrelated commentary instead of a proper move.',
            'score': 0.1
        },
        {
            'trajectory_id': '17',
            'explanation': 'Consistently used valid <move> tags and ended with a proper move, though didn’t reach 2048.',
            'score': 0.8
        },
        {
            'trajectory_id': '18',
            'explanation': 'Played a coherent sequence with valid <move> tags, ending correctly despite not achieving 2048.',
            'score': 0.8
        }
    ]
}
[RULER-DEBUG] Parsed 18 scores from model response.
[RULER-DEBUG] Trajectory updated with score=0.6, explanation=Chose a reasonable move (left) but used the wrong XML wrapper tag.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Did not issue a valid move in the required XML format.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Output is nonsensical and not a valid move.
[RULER-DEBUG] Trajectory updated with score=0.2, explanation=Made one valid <move>right</move> then failed to follow up with a proper move.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=No valid move in the required XML format.
[RULER-DEBUG] Trajectory updated with score=0.3, explanation=Recommended right but added extra text and did not stick to <move> XML.
[RULER-DEBUG] Trajectory updated with score=0.2, explanation=Offered a valid first move but then asked for more information instead of giving
next move.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Used an invalid direction (<move>east</move>), not one of the four allowed moves.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Never provided a correctly formatted move at the end.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Failed to supply a valid XML move.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Ended without a proper <move> XML response.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Stopped saying “No move” without providing a valid move XML.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Did not continue the game or give a final <move> XML.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Concluded with a message instead of a valid move tag.
[RULER-DEBUG] Trajectory updated with score=1.0, explanation=Successfully played through to 2048 and gave the correct final <move> XML.
[RULER-DEBUG] Trajectory updated with score=0.1, explanation=Provided many moves but ended with unrelated commentary instead of a proper move.
[RULER-DEBUG] Trajectory updated with score=0.8, explanation=Consistently used valid <move> tags and ended with a proper move, though didn’t
reach 2048.
[RULER-DEBUG] Trajectory updated with score=0.8, explanation=Played a coherent sequence with valid <move> tags, ending correctly despite not
achieving 2048.
[RULER-DEBUG] Returning new TrajectoryGroup with updated scores.
[DEBUG] TrajectoryGroup.__init__ started
[DEBUG] TrajectoryGroup.__init__ finished
[DEBUG] TrajectoryGroup.__init__ started
[DEBUG] TrajectoryGroup.__init__ finished
21:25:46 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: openai/o4-mini-2025-04-16
[2025-08-20 21:25:46] INFO cost_calculator.py:655: selected model name for cost calculation: openai/o4-mini-2025-04-16
[gather_trajectory_groups] after_each callback finished. Total groups now: 1
[gather_trajectory_groups] Finished.
Packed 18 trajectories into 2 sequences of length 4096
train:   0%|                                                                                                              | 0/2 [00:00<?, ?it/s]DEBUG put request: train ({'dir': '/workspace/verl/ART/.art/2048/models/tutorial-001/tensors', 'num_sequences': 2, 'sequence_length': 4096}, TrainConfig(learning_rate=1e-05, beta=0.0), {}, False) {}
[HANDLE_REQUEST] id=5ccbed5e-489b-4dd3-bb00-ed7a46247455, method=train, args=({'dir': '/workspace/verl/ART/.art/2048/models/tutorial-001/tensors', 'num_sequences': 2, 'sequence_length': 4096}, TrainConfig(learning_rate=1e-05, beta=0.0), {}, False), kwargs={}
[ASYNCGEN] Creating generator for train
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 10,000,000 | Num Epochs = 3 | Total steps = 10,000,002
O^O/ \_/ \    Batch size per device = 6 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 1 | Total batch size (6 x 1 x 1) = 6
 "-____-"     Trainable parameters = 4,399,104 of 498,431,872 (0.88% trained)
Unsloth: Will smartly offload gradients to save VRAM!
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 10,000,000 | Num Epochs = 3 | Total steps = 20,000,001
O^O/ \_/ \    Batch size per device = 3 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 1 | Total batch size (3 x 1 x 1) = 3
 "-____-"     Trainable parameters = 4,399,104 of 498,431,872 (0.88% trained)