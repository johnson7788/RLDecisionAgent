# 安装requirments.txt的特定版本的vllm
==((====))==  Unsloth 2025.8.6: Fast Qwen2_5_Vl patching. Transformers: 4.55.4. vLLM: 0.10.0.
   \\   /|    NVIDIA GeForce RTX 4090 D. Num GPUs = 3. Max memory: 23.546 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.1+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.3.1
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.31. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Traceback (most recent call last):
  File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 354, in <module>
    main()
  File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 350, in main
    train(args)
  File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 184, in train
    model, tokenizer = FastVisionModel.from_pretrained(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/unsloth/models/loader.py", line 840, in from_pretrained
    model, tokenizer = FastBaseModel.from_pretrained(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/unsloth/models/vision.py", line 444, in from_pretrained
    model = auto_model.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/models/auto/auto_factory.py", line 600, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py", line 317, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py", line 4999, in from_pretrained
    model = cls(config, *model_args, **model_kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Qwen2_5_VLForConditionalGeneration.__init__() got an unexpected keyword argument 'fast_inference'

# 报错，
RTX 4090D（SM 8.9）。在 vLLM 里，full CUDA graph 只在 FA3（FlashAttention 3）下支持，需要关掉full_cuda_graph，并清理缓存
export VLLM_COMPILATION_CONFIG='{"full_cuda_graph": false, "cudagraphs": false}'
export VLLM_ENFORCE_EAGER=1
rm -rf ~/.cache/vllm/torch_compile_cache/*
好像都不行，只能fast_inference=False了

python train_qwen_grpo.py   --dataset AI4Math/MathVista   --train_split testmini   --model_name ./unsloth/Qwen2.5-VL-3B-Instruct  --per_device_train_batch_size 1   --gradient_accumulation_steps 2   --output_dir outputs_qwen_vl_grpo
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
INFO 09-22 20:29:33 [__init__.py:244] Automatically detected platform cuda.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2025.9.7: Fast Qwen2_5_Vl patching. Transformers: 4.55.4. vLLM: 0.9.2.
   \\   /|    NVIDIA GeForce RTX 4090 D. Num GPUs = 1. Max memory: 23.546 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.0+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.3.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.30. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
INFO 09-22 20:30:54 [vllm_utils.py:688] Unsloth: Patching vLLM v1 graph capture
INFO 09-22 20:30:54 [vllm_utils.py:716] Unsloth: Patching vLLM v0 graph capture
Unsloth: Vision model detected, setting approx_max_num_seqs to 1
Unsloth: vLLM loading ./unsloth/Qwen2.5-VL-3B-Instruct with actual GPU utilization = 78.61%
Unsloth: Your GPU has CUDA compute capability 8.9 with VRAM = 23.55 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 2048. Num Sequences = 1.
Unsloth: vLLM's KV Cache can use up to 16.09 GB. Also swap space = 6 GB.
INFO 09-22 20:31:04 [config.py:841] This model supports multiple tasks: {'embed', 'generate', 'reward', 'classify'}. Defaulting to 'generate'.
INFO 09-22 20:31:04 [config.py:1472] Using max model len 16384
INFO 09-22 20:31:04 [config.py:2285] Chunked prefill is enabled with max_num_batched_tokens=16384.
Unsloth: vLLM Bitsandbytes config using kwargs = {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_quant_storage': 'uint8', 'bnb_4bit_quant_type': 'fp4', 'bnb_4bit_use_double_quant': False, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': [], 'llm_int8_threshold': 6.0}
INFO 09-22 20:31:05 [core.py:69] Initializing a V1 LLM engine (v0.9.2) with config: model='./unsloth/Qwen2.5-VL-3B-Instruct', speculative_config=None, tokenizer='./unsloth/Qwen2.5-VL-3B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=16384, download_dir=None, load_format=LoadFormat.BITSANDBYTES, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=./unsloth/Qwen2.5-VL-3B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"inductor","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"epilogue_fusion":true,"max_autotune":false,"shape_padding":true,"trace.enabled":false,"triton.cudagraphs":true,"debug":false,"dce":true,"memory_planning":true,"coordinate_descent_tuning":false,"trace.graph_diagram":false,"compile_threads":32,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.multi_kernel":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,"triton.cooperative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchmark_combo_kernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":true,"max_capture_size":512,"local_cache_dir":null}
INFO 09-22 20:31:06 [parallel_state.py:1076] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 09-22 20:31:06 [topk_topp_sampler.py:49] Using FlashInfer for top-p & top-k sampling.
INFO 09-22 20:31:06 [gpu_model_runner.py:1770] Starting to load model ./unsloth/Qwen2.5-VL-3B-Instruct...
INFO 09-22 20:31:07 [gpu_model_runner.py:1775] Loading model from scratch...
INFO 09-22 20:31:07 [cuda.py:284] Using Flash Attention backend on V1 engine.
INFO 09-22 20:31:07 [bitsandbytes_loader.py:499] Loading weights with BitsAndBytes quantization. May take a while ...
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.40s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.87s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.80s/it]

INFO 09-22 20:31:11 [punica_selector.py:19] Using PunicaWrapperGPU.
INFO 09-22 20:31:12 [gpu_model_runner.py:1801] Model loading took 2.5988 GiB and 4.162463 seconds
INFO 09-22 20:31:12 [gpu_model_runner.py:2238] Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum feature size.
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. Note that this behavior will be extended to all models in a future release.
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.
INFO 09-22 20:31:39 [backends.py:508] Using cache directory: /root/.cache/vllm/torch_compile_cache/14d375cdf9/rank_0_0/backbone for vLLM's torch.compile
INFO 09-22 20:31:39 [backends.py:519] Dynamo bytecode transform time: 14.73 s
Unsloth: Compiling kernels: 100%|████████| 18/18 [00:03<00:00,  5.42it/s, triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_15]
INFO 09-22 20:33:19 [backends.py:181] Cache the graph of shape None for later use
INFO 09-22 20:33:19 [backends.py:193] Compiling a graph for general shape takes 94.05 s
INFO 09-22 20:33:48 [monitor.py:34] torch.compile takes 108.77 s in total
INFO 09-22 20:33:56 [gpu_worker.py:232] Available KV cache memory: 13.48 GiB
INFO 09-22 20:33:57 [kv_cache_utils.py:716] GPU KV cache size: 392,576 tokens
INFO 09-22 20:33:57 [kv_cache_utils.py:720] Maximum concurrency for 16,384 tokens per request: 23.96x
[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/vllm_utils.py", line 1665, in load_vllm
[rank0]:     llm = LLM(**engine_args)
[rank0]:           ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/llm.py", line 271, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 501, in from_engine_args
[rank0]:     return engine_cls.from_vllm_config(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/llm_engine.py", line 124, in from_vllm_config
[rank0]:     return cls(vllm_config=vllm_config,
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/llm_engine.py", line 101, in __init__
[rank0]:     self.engine_core = EngineCoreClient.make_client(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 77, in make_client
[rank0]:     return InprocClient(vllm_config, executor_class, log_stats)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 230, in __init__
[rank0]:     self.engine_core = EngineCore(*args, **kwargs)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 82, in __init__
[rank0]:     self._initialize_kv_caches(vllm_config)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 169, in _initialize_kv_caches
[rank0]:     self.model_executor.initialize_from_config(kv_cache_configs)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 64, in initialize_from_config
[rank0]:     self.collective_rpc("initialize_from_config",
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 57, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py", line 2736, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py", line 601, in initialize_from_config
[rank0]:     self.worker.initialize_from_config(kv_cache_config)  # type: ignore
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 250, in initialize_from_config
[rank0]:     self.model_runner.initialize_kv_cache(kv_cache_config)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 2596, in initialize_kv_cache
[rank0]:     self.initialize_attn_backend(kv_cache_config)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 2366, in initialize_attn_backend
[rank0]:     attn_metadata_builder_i = attn_backend_i.get_builder_cls()(
[rank0]:                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_attn.py", line 181, in __init__
[rank0]:     raise ValueError(
[rank0]: ValueError: AoT scheduling is required for full cuda graph.

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 353, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 349, in main
[rank0]:     train(args)
[rank0]:   File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 182, in train
[rank0]:     model, tokenizer = FastVisionModel.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth/models/loader.py", line 881, in from_pretrained
[rank0]:     model, tokenizer = FastBaseModel.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth/models/vision.py", line 535, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/vllm_utils.py", line 1690, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError: AoT scheduling is required for full cuda graph.
[rank0]:[W922 20:34:04.426029050 ProcessGroupNCCL.cpp:1476] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

# 报错, 把 fast_inference 打开，然后 SamplingParams 就会报错
Map: 100%|███████████████████████████████████████████████████████████████████████████| 566/566 [01:38<00:00,  5.72 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████| 566/566 [00:00<00:00, 1616.85 examples/s]
`generation_config` default values have been modified to match model-specific defaults: {'temperature': 1e-06, 'repetition_penalty': 1.05, 'pad_token_id': 151654, 'bos_token_id': 151643, 'eos_token_id': [151645, 151643]}. If this is not desired, please set these values explicitly.
Traceback (most recent call last):
  File "/workspace/verl/docs/VL/qwen2_5_7b_vl_grpo.py", line 268, in <module>
    outputs = model.fast_generate(
              ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/generation/utils.py", line 2364, in generate
    generation_config, model_kwargs = self._prepare_generation_config(
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/generation/utils.py", line 1779, in _prepare_generation_config
    model_kwargs = generation_config.update(**kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'SamplingParams' object has no attribute 'update'


# pip uninstall -y flash-attn flash_attn
# 可选：明确禁用 xformers 对 flash-attn 的探测（双保险）
export XFORMERS_DISABLE_FLASH_ATTN=1

Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
INFO 09-23 09:19:37 [vllm_utils.py:688] Unsloth: Patching vLLM v1 graph capture
INFO 09-23 09:19:37 [vllm_utils.py:716] Unsloth: Patching vLLM v0 graph capture
Unsloth: Vision model detected, setting approx_max_num_seqs to 1
Unsloth: vLLM loading ./unsloth/Qwen2.5-VL-3B-Instruct with actual GPU utilization = 78.61%
Unsloth: Your GPU has CUDA compute capability 8.9 with VRAM = 23.55 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 2048. Num Sequences = 1.
Unsloth: vLLM's KV Cache can use up to 16.09 GB. Also swap space = 6 GB.
WARNING 09-23 09:19:37 [compilation.py:456] full_cuda_graph is deprecated, use cudagraph_mode=FULL instead.
Unsloth: Not an error, but `device` is not supported in vLLM. Skipping.
INFO 09-23 09:19:37 [utils.py:328] non-default args: {'load_format': 'bitsandbytes', 'dtype': torch.bfloat16, 'seed': 0, 'max_
model_len': 16384, 'enable_prefix_caching': True, 'swap_space': 6, 'gpu_memory_utilization': 0.7861018152974387, 'max_num_batc
hed_tokens': 16384, 'max_num_seqs': 1, 'max_logprobs': 0, 'disable_log_stats': True, 'quantization': 'bitsandbytes', 'limit_mm
_per_prompt': {'image': 1, 'video': 0}, 'enable_lora': True, 'max_lora_rank': 64, 'enable_chunked_prefill': True, 'compilation
_config': {"level":3,"debug_dump_path":"","cache_dir":"","backend":"inductor","custom_ops":[],"splitting_ops":null,"use_induct
or":true,"compile_sizes":null,"inductor_compile_config":{"epilogue_fusion":true,"max_autotune":false,"shape_padding":true,"tra
ce.enabled":false,"triton.cudagraphs":true,"debug":false,"dce":true,"memory_planning":true,"coordinate_descent_tuning":false,"
trace.graph_diagram":false,"compile_threads":32,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.m
ulti_kernel":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,
"triton.cooperative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchm
ark_combo_kernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"inductor_passes":{},
"cudagraph_mode":2,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":null,"cudagraph_copy_inputs":fa
lse,"full_cuda_graph":true,"pass_config":{},"max_capture_size":null,"local_cache_dir":null}, 'model': './unsloth/Qwen2.5-VL-3B
-Instruct'}
INFO 09-23 09:19:46 [__init__.py:742] Resolved architecture: Qwen2_5_VLForConditionalGeneration
INFO 09-23 09:19:46 [__init__.py:1815] Using max model len 16384
INFO 09-23 09:19:47 [scheduler.py:222] Chunked prefill is enabled with max_num_batched_tokens=16384.
WARNING 09-23 09:19:47 [lora.py:92] `lora_extra_vocab_size` is deprecated and will be removed in v0.12.0. Additional vocabular
y support for LoRA adapters is being phased out.
WARNING 09-23 09:19:47 [_ipex_ops.py:16] Import error msg: No module named 'intel_extension_for_pytorch'
_diagram":false,"compile_threads":32,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.multi_kernel
":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,"triton.coo
perative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchmark_combo_k
ernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_
mode":2,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[4,2,1],"cudagraph_copy_inputs":false,"ful
l_cuda_graph":true,"pass_config":{},"max_capture_size":4,"local_cache_dir":null}
[W923 09:19:48.138952968 ProcessGroupNCCL.cpp:981] Warning: TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environme
nt variable is thus deprecated. (function operator())
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
INFO 09-23 09:19:48 [parallel_state.py:1165] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 09-23 09:19:48 [topk_topp_sampler.py:58] Using FlashInfer for top-p & top-k sampling.
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved
 in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it aut
omatically. Loading from `preprocessor.json` will be removed in v5.0.
INFO 09-23 09:19:51 [gpu_model_runner.py:2338] Starting to load model ./unsloth/Qwen2.5-VL-3B-Instruct...
INFO 09-23 09:19:52 [gpu_model_runner.py:2370] Loading model from scratch...
WARNING 09-23 09:19:52 [cuda.py:217] Current `vllm-flash-attn` has a bug inside vision module, so we use xformers backend inst
ead. You can run `pip install flash-attn` to use flash-attention backend.
INFO 09-23 09:19:52 [cuda.py:362] Using Flash Attention backend on V1 engine.
INFO 09-23 09:19:52 [bitsandbytes_loader.py:758] Loading weights with BitsAndBytes quantization. May take a while ...
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.23it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.11s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]
INFO 09-23 09:19:54 [punica_selector.py:19] Using PunicaWrapperGPU.
INFO 09-23 09:19:55 [gpu_model_runner.py:2392] Model loading took 2.5987 GiB and 2.639683 seconds
INFO 09-23 09:19:55 [gpu_model_runner.py:3000] Encoder cache will be initialized with a budget of 16384 tokens, and profiled w
ith 1 image items of the maximum feature size.
[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/vllm_utils.py", line 1665, in load_vllm
[rank0]:     llm = LLM(**engine_args)
[rank0]:           ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/llm.py", line 282, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 493, in from_engine_args
[rank0]:     return engine_cls.from_vllm_config(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/llm_engine.py", line 134, in from_vllm_config
[rank0]:     return cls(vllm_config=vllm_config,
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/llm_engine.py", line 111, in __init__
[rank0]:     self.engine_core = EngineCoreClient.make_client(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 82, in make_client
[rank0]:     return InprocClient(vllm_config, executor_class, log_stats)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 245, in __init__
[rank0]:     self.engine_core = EngineCore(*args, **kwargs)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 91, in __init__
[rank0]:     self._initialize_kv_caches(vllm_config)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 183, in _initialize_kv_caches
[rank0]:     self.model_executor.determine_available_memory())
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 84, in determine_available_memory
[rank0]:     return self.collective_rpc("determine_available_memory")
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 58, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py", line 3060, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 120, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 263, in determine_available_memor
y
[rank0]:     self.model_runner.profile_run()
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 3017, in profile_run
[rank0]:     self.model.get_multimodal_embeddings(
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 1135, in get_multimod
al_embeddings
[rank0]:     vision_embeddings = self._process_image_input(multimodal_input)
[rank0]:                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 1059, in _process_ima
ge_input
[rank0]:     image_embeds = self.visual(pixel_values,
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 811, in forward
[rank0]:     hidden_states = blk(
[rank0]:                     ^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 455, in forward
[rank0]:     x_attn = self.attn(self.norm1(x),
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 398, in forward
[rank0]:     from xformers import ops as xops
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/xformers/ops/__init__.py", line 9, in <module>
[rank0]:     from .fmha import (
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/xformers/ops/fmha/__init__.py", line 10, in <module>
[rank0]:     from . import attn_bias, ck, ck_splitk, cutlass, flash, flash3, triton_splitk
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/xformers/ops/fmha/flash.py", line 67, in <module>
[rank0]:     import flash_attn
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/flash_attn/__init__.py", line 3, in <module>
[rank0]:     from flash_attn.flash_attn_interface import (
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/flash_attn/flash_attn_interface.py", line 15, in <module>
[rank0]:     import flash_attn_2_cuda as flash_attn_gpu
[rank0]: ImportError: /usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEa

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/docs/VL/qwen2_5_7b_vl_grpo.py", line 79, in <module>
[rank0]:     model, tokenizer = FastVisionModel.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth/models/loader.py", line 881, in from_pretrained
[rank0]:     model, tokenizer = FastBaseModel.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth/models/vision.py", line 535, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/vllm_utils.py", line 1690, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError: /usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEa
[rank0]:[W923 09:20:01.854118391 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())


# 代码报错, 改成load_in_4bit=False，也可能vllm版本问题
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 16.69it/s]

Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/vllm_utils.py", line 1665, in load_vllm
[rank0]:     llm = LLM(**engine_args)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py", line 271, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 501, in from_engine_args
[rank0]:     return engine_cls.from_vllm_config(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/llm_engine.py", line 124, in from_vllm_config
[rank0]:     return cls(vllm_config=vllm_config,
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/llm_engine.py", line 101, in __init__
[rank0]:     self.engine_core = EngineCoreClient.make_client(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core_client.py", line 77, in make_client
[rank0]:     return InprocClient(vllm_config, executor_class, log_stats)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core_client.py", line 230, in __init__
[rank0]:     self.engine_core = EngineCore(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core.py", line 75, in __init__
[rank0]:     self.model_executor = executor_class(vllm_config)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/executor_base.py", line 53, in __init__
[rank0]:     self._init_executor()
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py", line 48, in _init_executor
[rank0]:     self.collective_rpc("load_model")
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py", line 57, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/utils/__init__.py", line 2736, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_worker.py", line 185, in load_model
[rank0]:     self.model_runner.load_model()
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1776, in load_model
[rank0]:     self.model = model_loader.load_model(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 41, in load_model
[rank0]:     self.load_weights(model, model_config)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/bitsandbytes_loader.py", line 507, in load_weights
[rank0]:     loaded_weights = model.load_weights(qweight_iterator)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 1165, in load_weights
[rank0]:     return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 291, in load_weights
[rank0]:     autoloaded_weights = set(self._load_module("", self.module, weights))
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 249, in _load_module
[rank0]:     yield from self._load_module(prefix,
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 222, in _load_module
[rank0]:     loaded_params = module_load_weights(weights)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2.py", line 498, in load_weights
[rank0]:     return loader.load_weights(weights)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 291, in load_weights
[rank0]:     autoloaded_weights = set(self._load_module("", self.module, weights))
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 249, in _load_module
[rank0]:     yield from self._load_module(prefix,
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 222, in _load_module
[rank0]:     loaded_params = module_load_weights(weights)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2.py", line 420, in load_weights
[rank0]:     weight_loader(param, loaded_weight)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/linear.py", line 1282, in weight_loader
[rank0]:     assert param_data.shape == loaded_weight.shape
[rank0]: AssertionError

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/RLDecisionAgent/docs/VL/train_qwen_grpo.py", line 353, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/verl/RLDecisionAgent/docs/VL/train_qwen_grpo.py", line 349, in main
[rank0]:     train(args)
[rank0]:   File "/workspace/verl/RLDecisionAgent/docs/VL/train_qwen_grpo.py", line 182, in train
[rank0]:     model, tokenizer = FastVisionModel.from_pretrained(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth/models/loader.py", line 881, in from_pretrained
[rank0]:     model, tokenizer = FastBaseModel.from_pretrained(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth/models/vision.py", line 535, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/vllm_utils.py", line 1690, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:02<?, ?it/s]

[rank0]:[W924 21:07:37.702183097 ProcessGroupNCCL.cpp:1476] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())



# 报错，可能vllm版本问题, 升级vllm=0.10.2也不行
INFO 09-24 21:16:31 [gpu_model_runner.py:1770] Starting to load model unsloth/Qwen2.5-VL-3B-Instruct...
INFO 09-24 21:16:32 [gpu_model_runner.py:1775] Loading model from scratch...
[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/vllm_utils.py", line 1665, in load_vllm
[rank0]:     llm = LLM(**engine_args)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py", line 271, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 501, in from_engine_args
[rank0]:     return engine_cls.from_vllm_config(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/llm_engine.py", line 124, in from_vllm_config
[rank0]:     return cls(vllm_config=vllm_config,
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/llm_engine.py", line 101, in __init__
[rank0]:     self.engine_core = EngineCoreClient.make_client(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core_client.py", line 77, in make_client
[rank0]:     return InprocClient(vllm_config, executor_class, log_stats)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core_client.py", line 230, in __init__
[rank0]:     self.engine_core = EngineCore(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core.py", line 75, in __init__
[rank0]:     self.model_executor = executor_class(vllm_config)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/executor_base.py", line 53, in __init__
[rank0]:     self._init_executor()
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py", line 48, in _init_executor
[rank0]:     self.collective_rpc("load_model")
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py", line 57, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/utils/__init__.py", line 2736, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_worker.py", line 185, in load_model
[rank0]:     self.model_runner.load_model()
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1776, in load_model
[rank0]:     self.model = model_loader.load_model(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 38, in load_model
[rank0]:     model = initialize_model(vllm_config=vllm_config,
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/utils.py", line 64, in initialize_model
[rank0]:     return model_class(vllm_config=vllm_config, prefix=prefix)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 855, in __init__
[rank0]:     self.visual = Qwen2_5_VisionTransformer(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 542, in __init__
[rank0]:     self.blocks = nn.ModuleList([
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 543, in <listcomp>
[rank0]:     Qwen2_5_VisionBlock(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 376, in __init__
[rank0]:     self.attn = Qwen2_5_VisionAttention(embed_dim=dim,
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2_5_vl.py", line 255, in __init__
[rank0]:     raise RuntimeError(
[rank0]: RuntimeError: Qwen2.5-VL does not support _Backend.FLASHINFER backend now.

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/RLDecisionAgent/docs/VL/qwen2_5_7b_vl_grpo.py", line 79, in <module>
[rank0]:     model, tokenizer = FastVisionModel.from_pretrained(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth/models/loader.py", line 881, in from_pretrained
[rank0]:     model, tokenizer = FastBaseModel.from_pretrained(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth/models/vision.py", line 535, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/vllm_utils.py", line 1690, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError: Qwen2.5-VL does not support _Backend.FLASHINFER backend now.


# 不同的attention后端 export VLLM_ATTENTION_BACKEND=CUTLASS_MLA
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspace/verl/RLDecisionAgent/docs/VL/train_qwen_grpo.py", line 353, in <module>
    main()
  File "/workspace/verl/RLDecisionAgent/docs/VL/train_qwen_grpo.py", line 349, in main
    train(args)
  File "/workspace/verl/RLDecisionAgent/docs/VL/train_qwen_grpo.py", line 182, in train
    model, tokenizer = FastVisionModel.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/loader.py", line 881, in from_pretrained
    model, tokenizer = FastBaseModel.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/vision.py", line 535, in from_pretrained
    llm = load_vllm(**load_vllm_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/vllm_utils.py", line 1690, in load_vllm
    raise RuntimeError(error)
RuntimeError: Invalid attention backend: ''. Valid backends are: ['FLASH_ATTN', 'FLASH_ATTN_VLLM_V1', 'TRITON_ATTN_VLLM_V1', 'XFORMERS', 'ROCM_FLASH', 'ROCM_AITER_MLA', 'ROCM_AITER_MLA_VLLM_V1', 'ROCM_AITER_FA', 'TORCH_SDPA', 'TORCH_SDPA_VLLM_V1', 'FLASHINFER', 'FLASHINFER_VLLM_V1', 'FLASHINFER_MLA', 'TRITON_MLA', 'TRITON_MLA_VLLM_V1', 'CUTLASS_MLA', 'FLASHMLA', 'FLASHMLA_VLLM_V1', 'FLASH_ATTN_MLA', 'PALLAS', 'PALLAS_VLLM_V1', 'IPEX', 'DUAL_CHUNK_FLASH_ATTN', 'DIFFERENTIAL_FLASH_ATTN', 'NO_ATTENTION', 'FLEX_ATTENTION', 'TREE_ATTN', 'XFORMERS_VLLM_V1']