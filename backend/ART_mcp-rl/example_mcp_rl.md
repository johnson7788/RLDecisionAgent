# 示例
https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb#scrollTo=OsrwCDQ5cviC

# colab的显卡
Tesla T4 16GB

# 安装, 不要安装==0.3.11.post5版本
!uv pip install -q openpipe-art langchain-core tenacity "mcp>=1.11.0" "gql<4" aiohttp --no-cache-dir

# 配置smithery
https://smithery.ai/
配置
OPENROUTER_API_KEY = "sk-or-v1-xxx" 

打开https://smithery.ai/server/exa， 然后点击点击 Get URL with keys instead, 是网络搜索的工具
SMITHERY_MCP_URL = "https://server.smithery.ai/exa/mcp?api_key=552ddb78-0e95-4998-be87-b936502a5a97&profile=stiff-sole-4FpnEv"


# 运行

## Step0 运行llm_cache
```
cd RLDecisionAgent/backend
python LLM_cache.py
```

## Step1 数据准备
```
pip install --no-deps -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install weave polars torchtune trl unsloth apscheduler vllm -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install bitsandbytes langchain-core tenacity "mcp>=1.11.0" "gql<4"  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
cd backend/ART_mcp-rl
创建.env文件
# cat .env
OPENAI_API_KEY=sk-proj-x-xxxx
ALPHAVANTAGE_API_KEY=SGxxxx
WANDB_API_KEY=c93xxx
BALLDONTLIE_API_KEY=83xxx
# 复制一份到mcp_rl中
cp .env mcp_rl 

准备数据(其实已经存在数据了，在每个servers/xxx/scenarios/train.jsonl和val.jsonl)
python -m mcp_rl.scenario_generator servers/python/mcp_caculator/server_params.py
输出
weave: Please login to Weights & Biases (https://wandb.ai/) to continue...
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
wandb: No netrc file found, creating one.
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Currently logged in as: johnson- to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
weave: wandb version 0.21.1 is available!  To upgrade, please run:
weave:  $ pip install wandb --upgrade
weave: Logged in as Weights & Biases user: johnson-.
weave: View Weave data at https://wandb.ai/johnson-/mcp-agent-training/weave
Generating 24 scenarios using servers/python/mcp_alphavantage/server_params.py...
Output will be saved to: servers/python/mcp_alphavantage/scenarios/
Saved 16 training scenarios to servers/python/mcp_alphavantage/scenarios/train.jsonl
Saved 8 validation scenarios to servers/python/mcp_alphavantage/scenarios/val.jsonl

Generated 24 scenarios:
1. Task: Perform a comprehensive analysis by retrieving IBM's daily time series data, calculating both the SMA and RSI, then creating an in-depth report with a complete summary and analysis of the stock's trends.
   Difficulty: 5/5
2. Task: Calculate the 30-day Simple Moving Average (SMA) for Amazon (AMZN), compare it with the current stock price, and produce a thorough analysis and summary report on market trends.
   Difficulty: 2/5
3. Task: Search for companies with 'Google' in their name, retrieve related symbols, and produce a comprehensive report including a summary of company overviews and potential investment insights.
   Difficulty: 3/5
4. Task: Get the real-time stock quote for Nvidia (NVDA) and create a brief analysis report including a summary of its current market performance.
   Difficulty: 1/5
5. Task: Calculate the 14-day RSI for Alphabet (GOOG) on a daily interval, then generate a comprehensive analysis report that includes a summary of the market conditions and trends.
   Difficulty: 2/5
6. Task: Retrieve the weekly RSI for AMD, compile the indicator results, and develop a detailed trend analysis report that includes a comprehensive summary and market recommendations.
   Difficulty: 3/5
7. Task: Get the current stock price for Microsoft (MSFT) and generate a summary of the work done along with a thorough analysis and report of the real-time data.
   Difficulty: 1/5
8. Task: Perform a symbol search using the keyword 'energy' to identify potential investment opportunities in the energy sector, and produce a comprehensive report with analysis and summary of the identified stocks.
   Difficulty: 3/5
9. Task: Fetch the real-time stock quote for Tesla (TSLA) and compare it with its 30-day SMA to generate a detailed report with a summary and analysis of potential buy/sell signals.
   Difficulty: 3/5
10. Task: Retrieve daily time series data for Apple (AAPL) for the last 100 days, analyze the price trends, and generate a detailed summary and analysis report.
   Difficulty: 2/5
11. Task: Obtain the company overview for IBM and generate a report that includes a detailed summary and analysis of the company’s fundamental data.
   Difficulty: 1/5
12. Task: Perform a symbol search for companies in the pharmaceutical sector using the keyword 'pharma', then generate a report summarizing potential investment opportunities with detailed analysis and summary.
   Difficulty: 3/5
13. Task: Calculate the 14-day Relative Strength Index (RSI) for Netflix (NFLX) on a daily interval, then compile a comprehensive analysis report with a summary of the technical indicators.
   Difficulty: 2/5
14. Task: Calculate the SMA for Cisco (CSCO) using daily data, compare it with recent price movements, and generate a report with a thorough summary and technical analysis.
   Difficulty: 2/5
15. Task: Fetch the company overview for Facebook (META) and generate a detailed report that includes a thorough summary and analysis of the company’s fundamentals.
   Difficulty: 1/5
16. Task: Retrieve the company overview for McDonald's (MCD) and compile a report that includes a detailed summary and analysis of its fundamentals and market performance.
   Difficulty: 2/5
17. Task: Search for stock symbols using the keyword 'bank' to identify potential financial investments, and generate a detailed summary report that includes an analysis of each candidate.
   Difficulty: 3/5
18. Task: Get a real-time stock quote for Boeing (BA), retrieve its daily time series data for context, and develop an analysis report including a detailed summary of historical and current performance.
   Difficulty: 4/5
19. Task: Fetch the company overview for Twitter (TWTR) and generate a comprehensive report that includes a detailed summary and analysis of its fundamental health and market positioning.
   Difficulty: 1/5
20. Task: Fetch the real-time stock quote for Goldman Sachs (GS), calculate the 14-day RSI, and develop an integrated market analysis report with a thorough summary and recommendations.
   Difficulty: 4/5
21. Task: Search for companies in the retail sector, select one, retrieve its company overview and daily time series data, and compile a detailed analytical report with a thorough summary of its market outlook.
   Difficulty: 5/5
22. Task: Retrieve daily time series data for Intel (INTC), analyze historical price movements, and produce a detailed report including a summary of trends and insights.
   Difficulty: 3/5
23. Task: Gather daily time series data for Salesforce (CRM), compute the 30-day SMA, and prepare a comparative analysis report that includes a detailed summary of tech sector trends.
   Difficulty: 4/5
24. Task: Retrieve daily time series data for Oracle (ORCL) for the recent trading period, analyze the price movements and trends, and produce a comprehensive summary report.
   Difficulty: 3/5
```

## Step2 开始训练
```
wandb offline  # 关闭wandb上传
cd backend/ART_mcp-rl
export CUDA_VISIBLE_DEVICES=1,2
千万不要用http_proxy
python docs/ART/load_model.py
pip install polars torchtune trl unsloth apscheduler vllm fastapi-sso
# 安装一个依赖包
# 取消上传试验结果到s3
│ ✔  Edit examples/mcp-rl/mcp_rl/train.py:         await backend._experim... =>         # await backend._exper...        │
 │                                                                                                                        │
 │    170           print("starting train")                                                                               │
 │    171           await model.train(groups, config=art.TrainConfig(learning_rate=learning_rate))                        │
 │    172                                                                                                                 │
 │    173 -         await backend._experimental_push_to_s3(                                                               │
 │    174 -             model,                                                                                            │
 │    175 -         )                                                                                                     │
 │    173 +         # await backend._experimental_push_to_s3(                                                             │
 │    174 +         #     model,                                                                                          │
 │    175 +         # )                                                                                                   │
 │    176                                                                                                                 │
 │    177                                                                                                                 │
 │    178   def main():

# 运行命令
python -m mcp_rl.train --models=mcp-14b-alpha-001
输出:
weave: wandb version 0.21.1 is available!  To upgrade, please run:
weave:  $ pip install wandb --upgrade
weave: Logged in as Weights & Biases user: johnson-.
weave: View Weave data at https://wandb.ai/johnson-/mcp-agent-training/weave
Initializing Weave
Using model configuration: mcp-14b-alpha-001 (mcp-14b-alpha-001)
Starting MCP agent training..., use_skypilot: False
Loaded 16 training scenarios
Loaded 8 validation scenarios
Using config: max_turns=5, trajectories_per_group=7, groups_per_step=4, num_epochs=300, learning_rate=1e-06
wandb: Currently logged in as: johnson- to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
Task was destroyed but it is pending!
task: <Task cancelling name='Task-3' coro=<Event.wait() running at /usr/lib/python3.12/asyncio/locks.py:212> wait_for=<Future cancelled>>
wandb: Tracking run with wandb version 0.21.0
wandb: Run data is saved locally in /workspace/verl/RLDecisionAgent/ART/examples/mcp-rl/wandb/run-20250812_223105-mcp-14b-alpha-001
wandb: Run `wandb offline` to turn off syncing.
wandb: Resuming run mcp-14b-alpha-001
wandb: ⭐️ View project at https://wandb.ai/johnson-/mcp_alphavantage
wandb: 🚀 View run at https://wandb.ai/johnson-/mcp_alphavantage/runs/mcp-14b-alpha-001

weave: wandb version 0.21.1 is available!  To upgrade, please run:
weave:  $ pip install wandb --upgrade
weave: Logged in as Weights & Biases user: johnson-.
weave: View Weave data at https://wandb.ai/johnson-/mcp_alphavantage/weave
INFO 08-12 22:31:18 [__init__.py:235] Automatically detected platform cuda.
/workspace/verl/RLDecisionAgent/ART/src/art/__init__.py:10: UserWarning: WARNING: Unsloth should be imported before transformers, peft to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  import unsloth  # type: ignore # noqa: F401
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
INFO 08-12 22:31:31 [__init__.py:235] Automatically detected platform cuda.
Initializing Weave
Unsloth: Patching vLLM v1 graph capture
Unsloth: Patching vLLM v0 graph capture
==((====))==  Unsloth 2025.8.4: Fast Qwen2 patching. Transformers: 4.55.0. vLLM: 0.10.0.
   \\   /|    NVIDIA GeForce RTX 4090. Num GPUs = 1. Max memory: 47.499 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.1+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.3.1
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.31. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: vLLM loading unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit with actual GPU utilization = 78.28%
Unsloth: Your GPU has CUDA compute capability 8.9 with VRAM = 47.5 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 32768. Num Sequences = 320.
Unsloth: vLLM's KV Cache can use up to 36.71 GB. Also swap space = 0 GB.
Unsloth: Not an error, but `device` is not supported in vLLM. Skipping.
INFO 08-12 22:32:00 [config.py:1604] Using max model len 32768
Unsloth: vLLM Bitsandbytes config using kwargs = {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_quant_storage': 'uint8', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': ['lm_head', 'multi_modal_projector', 'merger', 'modality_projection', 'model.layers.0.self_attn', 'model.layers.0.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.21.mlp', 'model.layers.0.self_attn.q_proj'], 'llm_int8_threshold': 6.0}
INFO 08-12 22:32:01 [llm_engine.py:228] Initializing a V0 LLM engine (v0.10.0) with config: model='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', speculative_config=None, tokenizer='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.BITSANDBYTES, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit, num_scheduler_steps=16, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"inductor","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"epilogue_fusion":true,"max_autotune":false,"shape_padding":true,"trace.enabled":false,"triton.cudagraphs":true,"debug":false,"dce":true,"memory_planning":true,"coordinate_descent_tuning":true,"trace.graph_diagram":false,"compile_threads":8,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.multi_kernel":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,"triton.cooperative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchmark_combo_kernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":320,"local_cache_dir":null}, use_cached_outputs=False,
INFO 08-12 22:32:03 [cuda.py:398] Using Flash Attention backend.
INFO 08-12 22:32:04 [parallel_state.py:1102] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 08-12 22:32:04 [model_runner.py:1083] Starting to load model unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit...
INFO 08-12 22:32:04 [bitsandbytes_loader.py:733] Loading weights with BitsAndBytes quantization. May take a while ...
INFO 08-12 22:32:06 [weight_utils.py:296] Using model weights format ['*.safetensors']
INFO 08-12 22:32:07 [weight_utils.py:312] Time spent downloading weights for unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit: 0.581658 seconds
INFO 08-12 22:32:07 [weight_utils.py:349] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 40.98it/s]

Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.59it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.59it/s]

INFO 08-12 22:32:08 [punica_selector.py:19] Using PunicaWrapperGPU.
INFO 08-12 22:32:09 [model_runner.py:1115] Model loading took 0.5151 GiB and 3.479114 seconds
INFO 08-12 22:32:11 [worker.py:295] Memory profiling takes 1.73 seconds
INFO 08-12 22:32:11 [worker.py:295] the current vLLM instance can use total_gpu_memory (47.50GiB) x gpu_memory_utilization (0.78) = 37.18GiB
INFO 08-12 22:32:11 [worker.py:295] model weights take 0.52GiB; non_torch_memory takes 0.08GiB; PyTorch activation peak memory takes 1.78GiB; the rest of the memory reserved for KV Cache is 34.81GiB.
INFO 08-12 22:32:11 [executor_base.py:113] # cuda blocks: 190100, # CPU blocks: 0
INFO 08-12 22:32:11 [executor_base.py:118] Maximum concurrency for 32768 tokens per request: 92.82x
INFO 08-12 22:32:11 [vllm_utils.py:669] Unsloth: Running patched vLLM v0 `capture_model`.
INFO 08-12 22:32:11 [model_runner.py:1385] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|████████████████████████████████████████████████████████████████████████████| 43/43 [00:07<00:00,  5.55it/s]
INFO 08-12 22:32:19 [model_runner.py:1537] Graph capturing finished in 8 secs, took 0.60 GiB
INFO 08-12 22:32:19 [vllm_utils.py:676] Unsloth: Patched vLLM v0 graph capture finished in 8 secs.
INFO 08-12 22:32:20 [llm_engine.py:424] init engine (profile, create kv cache, warmup model) took 11.33 seconds
Unsloth: Just some info: will skip parsing ['pre_feedforward_layernorm', 'k_norm', 'post_feedforward_layernorm', 'q_norm']
Unsloth: Just some info: will skip parsing ['pre_feedforward_layernorm', 'k_norm', 'post_feedforward_layernorm', 'q_norm']
Unsloth 2025.8.4 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers.
[2025-08-12 22:32:44,519] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-08-12 22:32:45,053] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False

```

