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

# 报错，关掉fast_inference
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

# 报错, 把 fast_inference 打开，然后 SamplingParams 就是正确的类型。
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