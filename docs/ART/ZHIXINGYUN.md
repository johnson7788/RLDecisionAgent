# RTX 4090 48GB

## 智星云上的Huggingface镜像(不好用)
export HF_ENDPOINT=http://192.168.50.202:18090

##  尝试使用Areal的镜像
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name areal ghcr.io/inclusionai/areal-runtime:v0.3.0.post2 sleep infinity
docker start areal
docker exec -it areal bash
# 设置pip镜像源
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
cd ART
pip install .
pip install ".[backend]"
# 设置代理，安装git上的项目
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPs_PROXY=http://127.0.0.1:7890
pip install 'torchtune @ git+https://github.com/pytorch/torchtune.git'
pip install 'unsloth-zoo @ git+https://github.com/bradhilton/unsloth-zoo'


## 尝试安装
https://art.openpipe.ai/getting-started/installation-setup
pip install openpipe-art openpipe-art[backend] -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


## 测试cli端能过托管模型,先修改ART/src/art/cli.py，添加
if __name__ == "__main__":
    app()

然后运行下面的客户端，最后运行请求代码python docs/ART/art_model_load.py，会输出
python -m art.cli
Starting the server...
INFO:     Started server process [178]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7999 (Press CTRL+C to quit)
INFO 08-16 20:00:31 [__init__.py:235] Automatically detected platform cuda.
_prepare_backend_for_training启动OpenAI服务: None
/workspace/verl/RLDecisionAgent/ART/src/art/__init__.py:10: UserWarning: WARNING: Unsloth should be imported before transformers, peft to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  import unsloth  # type: ignore # noqa: F401
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
INFO 08-16 20:00:46 [__init__.py:235] Automatically detected platform cuda.
Unsloth: Patching vLLM v1 graph capture
Unsloth: Patching vLLM v0 graph capture
==((====))==  Unsloth 2025.8.4: Fast Qwen2 patching. Transformers: 4.55.1. vLLM: 0.10.0.
   \\   /|    NVIDIA GeForce RTX 3090. Num GPUs = 1. Max memory: 23.684 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.1+cu126. CUDA: 8.6. CUDA Toolkit: 12.6. Triton: 3.3.1
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.31. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: vLLM loading unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit with actual GPU utilization = 77.99%
Unsloth: Your GPU has CUDA compute capability 8.6 with VRAM = 23.68 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 32768. Num Sequences = 288.
Unsloth: vLLM's KV Cache can use up to 18.0 GB. Also swap space = 0 GB.
Unsloth: Not an error, but `device` is not supported in vLLM. Skipping.
INFO 08-16 20:01:15 [config.py:1604] Using max model len 32768
Unsloth: vLLM Bitsandbytes config using kwargs = {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_quant_storage': 'uint8', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': ['lm_head', 'multi_modal_projector', 'merger', 'modality_projection', 'model.layers.0.self_attn', 'model.layers.0.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.21.mlp', 'model.layers.0.self_attn.q_proj'], 'llm_int8_threshold': 6.0}
INFO 08-16 20:01:16 [llm_engine.py:228] Initializing a V0 LLM engine (v0.10.0) with config: model='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', speculative_config=None, tokenizer='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.BITSANDBYTES, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit, num_scheduler_steps=16, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"inductor","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"epilogue_fusion":true,"max_autotune":false,"shape_padding":true,"trace.enabled":false,"triton.cudagraphs":true,"debug":false,"dce":true,"memory_planning":true,"coordinate_descent_tuning":true,"trace.graph_diagram":false,"compile_threads":8,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.multi_kernel":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,"triton.cooperative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchmark_combo_kernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":288,"local_cache_dir":null}, use_cached_outputs=False,
INFO 08-16 20:01:19 [cuda.py:398] Using Flash Attention backend.
INFO 08-16 20:01:19 [parallel_state.py:1102] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 08-16 20:01:19 [model_runner.py:1083] Starting to load model unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit...
INFO 08-16 20:01:20 [bitsandbytes_loader.py:733] Loading weights with BitsAndBytes quantization. May take a while ...
INFO 08-16 20:01:22 [weight_utils.py:296] Using model weights format ['*.safetensors']
INFO 08-16 20:01:23 [weight_utils.py:312] Time spent downloading weights for unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit: 0.727459 seconds
INFO 08-16 20:01:24 [weight_utils.py:349] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.55it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.55it/s]

Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.45s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.45s/it]

INFO 08-16 20:01:25 [punica_selector.py:19] Using PunicaWrapperGPU.
INFO 08-16 20:01:26 [model_runner.py:1115] Model loading took 0.5151 GiB and 5.839312 seconds
INFO 08-16 20:01:30 [worker.py:295] Memory profiling takes 3.47 seconds
INFO 08-16 20:01:30 [worker.py:295] the current vLLM instance can use total_gpu_memory (23.68GiB) x gpu_memory_utilization (0.78) = 18.47GiB
INFO 08-16 20:01:30 [worker.py:295] model weights take 0.52GiB; non_torch_memory takes 0.06GiB; PyTorch activation peak memory takes 1.61GiB; the rest of the memory reserved for KV Cache is 16.29GiB.
INFO 08-16 20:01:31 [executor_base.py:113] # cuda blocks: 88952, # CPU blocks: 0
INFO 08-16 20:01:31 [executor_base.py:118] Maximum concurrency for 32768 tokens per request: 43.43x
INFO 08-16 20:01:31 [vllm_utils.py:669] Unsloth: Running patched vLLM v0 `capture_model`.
INFO 08-16 20:01:31 [model_runner.py:1385] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|██████████████████████████████████████████████████████████████████████████████| 39/39 [00:07<00:00,  5.14it/s]
INFO 08-16 20:01:38 [model_runner.py:1537] Graph capturing finished in 8 secs, took 0.55 GiB
INFO 08-16 20:01:38 [vllm_utils.py:676] Unsloth: Patched vLLM v0 graph capture finished in 8 secs.
INFO 08-16 20:01:39 [llm_engine.py:424] init engine (profile, create kv cache, warmup model) took 12.81 seconds
Unsloth: Just some info: will skip parsing ['post_feedforward_layernorm', 'pre_feedforward_layernorm', 'q_norm', 'k_norm']
Unsloth: Just some info: will skip parsing ['post_feedforward_layernorm', 'pre_feedforward_layernorm', 'q_norm', 'k_norm']
Unsloth 2025.8.4 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers.
[2025-08-16 20:02:00,524] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-08-16 20:02:01,573] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
INFO:     127.0.0.1:44172 - "POST /_prepare_backend_for_training HTTP/1.1" 200 OK

请求端请求时会输出
python art_model_load.py
正在请求后端服务器加载模型: my-qwen-test (base: Qwen/Qwen2.5-0.5B-Instruct)
这可能需要一些时间...

✅ 后端成功加载模型！
   耗时: 98.75 秒
   模型现在被托管在: http://0.0.0.0:8000/v1


## 直接测试vllm是否正常，并且使用openai协议连接vllm的代码是否正常
python test_vllm_server.py

## 测试huggingface连通和模型使用
python load_model.py


## 开机自动启动
docker update verl --restart always
docker update wandb-local --restart always
[auto_start_clash.md](..%2F..%2Ftools%2Fauto_start_clash.md)
[LLMcache_service.md](..%2F..%2Ftools%2FLLMcache_service.md)