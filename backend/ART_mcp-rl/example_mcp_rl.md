# 示例
https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb#scrollTo=OsrwCDQ5cviC

# colab的显卡
Tesla T4 16GB

# 安装, 不要安装==0.3.11.post5版本
!uv pip install -q openpipe-art langchain-core tenacity "mcp>=1.11.0" "gql<4" aiohttp --no-cache-dir

# 配置smithery
https://smithery.ai/

打开https://smithery.ai/server/exa， 然后点击点击 Get URL with keys instead, 是网络搜索的工具
SMITHERY_MCP_URL = "https://server.smithery.ai/exa/mcp?api_key=552ddb78-0e95-4998-be87-b936502a5a97&profile=stiff-sole-4FpnEv"

# 测试使用模型进行评分
```
# 启动缓存
cd RLDecisionAgent/backend/ART_mcp-rl
python LLM_cache.py
# 启动测试
python -m mcp_rl.benchmarks.generate_benchmarks

```

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
export HF_ENDPOINT=https://hf-mirror.com
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


# 请求大模型的提示词 o3, Ruler模型
Prompt:
所有下面的轨迹都有相同的目标。你的任务是对每一个轨迹进行评估，并给它们一个 0 到 1 之间的分数。  
在评分时请根据你对智能体目标的最佳判断。  

评分标准：  
- 能够实现目标的轨迹，一定要比没有实现目标的轨迹得到显著更高的分数。  
- 实现目标更高效（例如避免无效的绕路）的轨迹，要比低效实现目标的轨迹得到更高分数。  
- 如果一个轨迹只是稍微比另一个好一点，那么分数差距应该很小；如果明显更好，那么分数差距应该很大。  
- 对于没有完成目标但朝着目标有所进展的轨迹，可以给予部分分数。  


Context:
[{
 "content": "你是一个 MCP (Model Context Protocol) agent。你可以通过服务器访问 MCP 工具。使用这些工具来完成任务。你总共有 5 次回合。只允许使用工具调用。完成任务时请调用 'complete_task'。",
 "role": "system"
}, {
 "content": "请完成以下任务：1425 + 73410。",
 "role": "user"
}]

trajectory: 
轨迹：

轨迹 1
助手调用 add 工具，得到结果 74835，最后回答：“1425 和 73410 的和是 74835。”

轨迹 2
同轨迹 1，调用工具并正确输出答案。

轨迹 3
同轨迹 1，调用工具并正确输出答案。

轨迹 4
同样调用工具并正确输出，但最后回答用的是“The total of ...” 而不是“The sum of ...”。

轨迹 5
同轨迹 1，调用工具并正确输出答案。

轨迹 6
同轨迹 1，调用工具并正确输出答案。

轨迹 7
调用工具并正确输出答案，但最后回答稍微多了一句：“现在你得到了结果：74835。今天还需要我帮你做其他事吗？”

要求Ruler模型的输出：
返回格式说明（要求模型输出）：

TrajectoryScore（单个轨迹的评分结果）：

trajectory_id: 轨迹 ID

explanation: 对该轨迹表现的简短说明

score: 一个 0 到 1 之间的分数

Response（整体输出）：
包含一个 scores 数组，每个元素都是一个 TrajectoryScore。

LLM返回返回的评分结果类似如下
{
  "scores": [
    {
      "trajectory_id": "1",
      "explanation": "这个轨迹是空的，没有尝试进行计算，因此没有实现目标。",
      "score": 0.0
    },
    {
      "trajectory_id": "2",
      "explanation": "这个轨迹正确地使用除法工具计算了 (100 - 20) / (2 * 2) = 20.0，高效地完成了目标。",
      "score": 1.0
    }
  ]
}


# mcp_rl_test.py，训练后的模型测试结果
python mcp_rl_test.py
Registering model...，进行super中..
调用backend的_prepare_backend_for_training
INFO 08-19 20:44:04 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 08-19 20:44:04 [__init__.py:239] Automatically detected platform cuda.
⚙️  Running in WANDB offline mode
_prepare_backend_for_training启动OpenAI服务: None
[start_openai_server] 开始启动 OpenAI server...
[start_openai_server] 当前 output_dir: /workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001
[start_openai_server] 传入 config: None
[start_openai_server] get_last_checkpoint_dir 返回: /workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/checkpoints/0008
[start_openai_server] 停止可能已有的 OpenAI server...
[start_openai_server] 旧的 OpenAI server 已停止
[start_openai_server] 准备启动新的 openai_server_task，配置如下：
  - model_name: mcp-14b-alpha-001
  - base_model: Qwen/Qwen2.5-0.5B-Instruct
  - log_file: /workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/logs/vllm.log
  - lora_path: /workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/checkpoints/0008
  - config: {'log_file': '/workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/logs/vllm.log', 'server_args': {'api_key': 'default', 'lora_modules': ['{"name": "mcp-14b-alpha-001", "path": "/workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/checkpoints/0008"}'], 'return_tokens_as_token_ids': True, 'enable_auto_tool_choice': True, 'tool_call_parser': 'hermes'}, 'engine_args': {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'num_scheduler_steps': 16, 'served_model_name': 'Qwen/Qwen2.5-0.5B-Instruct', 'disable_log_requests': True, 'generation_config': 'vllm'}}
/workspace/verl/ART/src/art/unsloth/state.py:11: UserWarning: WARNING: Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  import unsloth  # type: ignore
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Unsloth: Patching vLLM v1 graph capture
Unsloth: Patching vLLM v0 graph capture
==((====))==  Unsloth 2025.8.6: Fast Qwen2 patching. Transformers: 4.55.2. vLLM: 0.8.5.post1.
   \\   /|    NVIDIA GeForce RTX 4090 D. Num GPUs = 2. Max memory: 23.546 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.6.0+cu124. CUDA: 8.9. CUDA Toolkit: 12.4. Triton: 3.2.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post2. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: vLLM loading unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit with actual GPU utilization = 17.34%
Unsloth: Your GPU has CUDA compute capability 8.9 with VRAM = 23.55 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 32768. Num Sequences = 160.
Unsloth: vLLM's KV Cache can use up to 3.61 GB. Also swap space = 6 GB.
INFO 08-19 20:44:45 [config.py:717] This model supports multiple tasks: {'generate', 'score', 'classify', 'reward', 'embed'}. Defaulting to 'generate'.
Unsloth: vLLM Bitsandbytes config using kwargs = {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_quant_storage': 'uint8', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': ['lm_head', 'multi_modal_projector', 'merger', 'modality_projection', 'model.layers.0.self_attn', 'model.layers.0.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.21.mlp', 'model.layers.0.self_attn.q_proj'], 'llm_int8_threshold': 6.0}
INFO 08-19 20:44:45 [llm_engine.py:240] Initializing a V0 LLM engine (v0.8.5.post1) with config: model='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', speculative_config=None, tokenizer='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=bitsandbytes, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='auto', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit, num_scheduler_steps=16, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"level":0,"backend":"inductor","splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"epilogue_fusion":true,"max_autotune":false,"shape_padding":true,"trace.enabled":false,"triton.cudagraphs":true,"debug":false,"dce":true,"memory_planning":true,"coordinate_descent_tuning":true,"trace.graph_diagram":false,"compile_threads":32,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.multi_kernel":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,"triton.cooperative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchmark_combo_kernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":160}, use_cached_outputs=False,
INFO 08-19 20:44:50 [cuda.py:292] Using Flash Attention backend.
INFO 08-19 20:45:00 [parallel_state.py:1004] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0
INFO 08-19 20:45:00 [model_runner.py:1108] Starting to load model unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit...
INFO 08-19 20:45:00 [loader.py:1187] Loading weights with BitsAndBytes quantization. May take a while ...
INFO 08-19 20:45:09 [weight_utils.py:265] Using model weights format ['*.safetensors']
INFO 08-19 20:45:10 [weight_utils.py:281] Time spent downloading weights for unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit: 0.915206 seconds
INFO 08-19 20:45:10 [weight_utils.py:315] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.04it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.04it/s]

Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.70it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.70it/s]

INFO 08-19 20:45:11 [punica_selector.py:18] Using PunicaWrapperGPU.
INFO 08-19 20:45:12 [model_runner.py:1140] Model loading took 0.5153 GiB and 10.813045 seconds
INFO 08-19 20:45:15 [worker.py:287] Memory profiling takes 2.76 seconds
INFO 08-19 20:45:15 [worker.py:287] the current vLLM instance can use total_gpu_memory (23.55GiB) x gpu_memory_utilization (0.17) = 4.08GiB
INFO 08-19 20:45:15 [worker.py:287] model weights take 0.52GiB; non_torch_memory takes 0.07GiB; PyTorch activation peak memory takes 1.18GiB; the rest of the memory reserved for KV Cache is 2.32GiB.
INFO 08-19 20:45:15 [executor_base.py:112] # cuda blocks: 12652, # CPU blocks: 32768
INFO 08-19 20:45:15 [executor_base.py:117] Maximum concurrency for 32768 tokens per request: 6.18x
INFO 08-19 20:45:19 [vllm_utils.py:671] Unsloth: Running patched vLLM v0 `capture_model`.
INFO 08-19 20:45:19 [model_runner.py:1450] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|██████████████████████████████████████████████████████████████████| 23/23 [00:07<00:00,  2.97it/s]
INFO 08-19 20:45:27 [model_runner.py:1592] Graph capturing finished in 8 secs, took 0.35 GiB
INFO 08-19 20:45:27 [vllm_utils.py:678] Unsloth: Patched vLLM v0 graph capture finished in 8 secs.
INFO 08-19 20:45:27 [llm_engine.py:437] init engine (profile, create kv cache, warmup model) took 15.92 seconds
Unsloth: Just some info: will skip parsing ['k_norm', 'q_norm', 'post_feedforward_layernorm', 'pre_feedforward_layernorm']
Unsloth: Just some info: will skip parsing ['k_norm', 'q_norm', 'post_feedforward_layernorm', 'pre_feedforward_layernorm']
Unsloth 2025.8.6 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers.
Unsloth: Already have LoRA adapters! We shall skip this step.
[openai_server_task] 初始化开始...
[openai_server_task] 调用 subclass_chat_completion_request()...
[openai_server_task] 导入 vllm.entrypoints.openai.api_server...
[openai_server_task] 应用 vLLM patches...
[openai_server_task] 设置 vLLM 日志文件: /workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/logs/vllm.log
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
[2025-08-19 20:45:40] INFO _client.py:1740: HTTP Request: GET http://0.0.0.0:8000/v1/models "HTTP/1.1 200 OK"
[openai_server_task.test_client] 成功连通 server！
[openai_server_task] task <Task finished name='Task-3' coro=<openai_server_task.<locals>.test_client() done, defined at /workspace/verl/ART/src/art/vllm/server.py:98> result=None> 已完成，检查结果...
[openai_server_task] server 成功启动！
[start_openai_server] openai_server_task 启动完成: <Task pending name='Task-2' coro=<run_server() running at /usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/api_server.py:1094> wait_for=<Task pending name='Task-5' coro=<Server.serve() running at /usr/local/lib/python3.10/dist-packages/uvicorn/server.py:70> wait_for=<Future pending cb=[Task.__wakeup()]> cb=[Task.__wakeup()]>>
[start_openai_server] 设置 lora: /workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/checkpoints/0008
[start_openai_server] 启动流程完成 ✅

测试： 1: Calculate (10 - 9) * 3.
[INFO] 开始 rollout，任务: Calculate (10 - 9) * 3., 最大轮数: 10
[INFO] 连接 MCP 服务器...
[INFO] MCP 服务器连接成功，初始化 session
[INFO] MCP server 返回 4 个工具
[DEBUG] Available MCP tools: ['add', 'subtract', 'multiply', 'divide']
[DEBUG] 初始消息: [{'role': 'system', 'content': "You are an MCP (Model Context Protocol) agent.\nYou have access to MCP tools through the server. Use them to complete your task.\nYou have a total of 10 turns. Only use tool calls.\nCall 'complete_task' when finished."}, {'role': 'user', 'content': 'Please complete this task: Calculate (10 - 9) * 3.'}]
[INFO] 开始第 1 轮交互
[2025-08-19 20:45:41] INFO _client.py:1740: HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
[DEBUG] LLM Choice: ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='chatcmpl-tool-4f1d268e3c554e75bc18947c927fd86c', function=Function(arguments='{"a": 1, "b": 3}', name='multiply'), type='function')], reasoning_content=None)
[INFO] 执行工具调用: multiply, args={'a': 1, 'b': 3}
[INFO] 工具 multiply 调用返回长度: 1
[DEBUG] 工具调用结果: 3...
[INFO] 开始第 2 轮交互
[2025-08-19 20:45:41] INFO _client.py:1740: HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
[DEBUG] LLM Choice: ChatCompletionMessage(content='The result of the calculation (10 - 9) * 3 is 3.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content=None)
[INFO] 本轮无工具调用, 继续下一轮或结束
[INFO] rollout 结束, 总轮数: 2, 完成状态: False
[DEBUG] 最终消息列表:
{'role': 'system', 'content': "You are an MCP (Model Context Protocol) agent.\nYou have access to MCP tools through the server. Use them to complete your task.\nYou have a total of 10 turns. Only use tool calls.\nCall 'complete_task' when finished."}
{'role': 'user', 'content': 'Please complete this task: Calculate (10 - 9) * 3.'}
Choice(finish_reason='tool_calls', index=0, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='token_id:151657', bytes=[60, 116, 111, 111, 108, 95, 99, 97, 108, 108, 62], logprob=-0.0047868178226053715, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:198', bytes=[10], logprob=-0.0008794969180598855, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:4913', bytes=[123, 34], logprob=-0.0005046047735959291, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:606', bytes=[110, 97, 109, 101], logprob=-2.95634672511369e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:788', bytes=[34, 58], logprob=-5.960462772236497e-07, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:330', bytes=[32, 34], logprob=-0.00018368464952800423, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:64648', bytes=[109, 117, 108, 116, 105, 112, 108, 121], logprob=-0.0758897215127945, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:497', bytes=[34, 44], logprob=-1.823885577323381e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:330', bytes=[32, 34], logprob=-2.50339189733495e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:16370', bytes=[97, 114, 103, 117, 109, 101, 110, 116, 115], logprob=-0.0003599472693167627, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:788', bytes=[34, 58], logprob=-9.846202738117427e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:5212', bytes=[32, 123, 34], logprob=-0.004258967936038971, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:64', bytes=[97], logprob=-2.3603161025675945e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:788', bytes=[34, 58], logprob=-9.536738616588991e-07, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-0.053150810301303864, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:16', bytes=[49], logprob=-0.008982841856777668, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:11', bytes=[44], logprob=-2.930082321166992, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:330', bytes=[32, 34], logprob=-1.6689286894688848e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:65', bytes=[98], logprob=-1.311301275563892e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:788', bytes=[34, 58], logprob=-7.152555099310121e-07, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-1.966933996300213e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:18', bytes=[51], logprob=-0.0018358058296144009, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:11248', bytes=[125, 125, 10], logprob=-0.004112596623599529, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:151658', bytes=[60, 47, 116, 111, 111, 108, 95, 99, 97, 108, 108, 62], logprob=-0.002861098386347294, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:151645', bytes=[], logprob=-0.00021610308613162488, top_logprobs=[])], refusal=None), message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='chatcmpl-tool-4f1d268e3c554e75bc18947c927fd86c', function=Function(arguments='{"a": 1, "b": 3}', name='multiply'), type='function')], reasoning_content=None), stop_reason=None)
{'role': 'tool', 'tool_call_id': 'chatcmpl-tool-4f1d268e3c554e75bc18947c927fd86c', 'content': '3'}
Choice(finish_reason='stop', index=0, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='token_id:785', bytes=[84, 104, 101], logprob=-0.020876435562968254, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:1102', bytes=[32, 114, 101, 115, 117, 108, 116], logprob=-0.04908208176493645, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:315', bytes=[32, 111, 102], logprob=-0.0013579442165791988, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:279', bytes=[32, 116, 104, 101], logprob=-2.5378024578094482, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:21937', bytes=[32, 99, 97, 108, 99, 117, 108, 97, 116, 105, 111, 110], logprob=-0.489208459854126, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:320', bytes=[32, 40], logprob=-0.08190760016441345, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:16', bytes=[49], logprob=-9.858122211880982e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-7.748573807475623e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:481', bytes=[32, 45], logprob=-0.0008654424455016851, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-5.960446742392378e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:24', bytes=[57], logprob=-9.059865078597795e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:8', bytes=[41], logprob=-0.0002803409588523209, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:353', bytes=[32, 42], logprob=-4.51792984677013e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-1.6689286894688848e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:18', bytes=[51], logprob=-5.960462772236497e-07, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:374', bytes=[32, 105, 115], logprob=-0.0003819928097072989, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-0.00017975145601667464, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:18', bytes=[51], logprob=-0.001562089892104268, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:13', bytes=[46], logprob=-0.02700188383460045, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:151645', bytes=[], logprob=-0.0893554538488388, top_logprobs=[])], refusal=None), message=ChatCompletionMessage(content='The result of the calculation (10 - 9) * 3 is 3.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content=None), stop_reason=None)
模型输出: The result of the calculation (10 - 9) * 3 is 3.

测试： 2: Calculate 400 / (10 + 10).
[INFO] 开始 rollout，任务: Calculate 400 / (10 + 10)., 最大轮数: 10
[INFO] 连接 MCP 服务器...
[INFO] MCP 服务器连接成功，初始化 session
[INFO] MCP server 返回 4 个工具
[DEBUG] Available MCP tools: ['add', 'subtract', 'multiply', 'divide']
[DEBUG] 初始消息: [{'role': 'system', 'content': "You are an MCP (Model Context Protocol) agent.\nYou have access to MCP tools through the server. Use them to complete your task.\nYou have a total of 10 turns. Only use tool calls.\nCall 'complete_task' when finished."}, {'role': 'user', 'content': 'Please complete this task: Calculate 400 / (10 + 10).'}]
[INFO] 开始第 1 轮交互
[2025-08-19 20:45:42] INFO _client.py:1740: HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
[DEBUG] LLM Choice: ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='chatcmpl-tool-a8c24beba5ac4bdf8bd97bc31630e30d', function=Function(arguments='{"a": 400, "b": 20}', name='divide'), type='function')], reasoning_content=None)
[INFO] 执行工具调用: divide, args={'a': 400, 'b': 20}
[INFO] 工具 divide 调用返回长度: 4
[DEBUG] 工具调用结果: 20.0...
[INFO] 开始第 2 轮交互
[2025-08-19 20:45:43] INFO _client.py:1740: HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
[DEBUG] LLM Choice: ChatCompletionMessage(content='The result of 400 / (10 + 10) is 20.0.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content=None)
[INFO] 本轮无工具调用, 继续下一轮或结束
[INFO] rollout 结束, 总轮数: 2, 完成状态: False
[DEBUG] 最终消息列表:
{'role': 'system', 'content': "You are an MCP (Model Context Protocol) agent.\nYou have access to MCP tools through the server. Use them to complete your task.\nYou have a total of 10 turns. Only use tool calls.\nCall 'complete_task' when finished."}
{'role': 'user', 'content': 'Please complete this task: Calculate 400 / (10 + 10).'}
Choice(finish_reason='tool_calls', index=0, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='token_id:151657', bytes=[60, 116, 111, 111, 108, 95, 99, 97, 108, 108, 62], logprob=-0.003934025764465332, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:198', bytes=[10], logprob=-0.0004415729199536145, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:4913', bytes=[123, 34], logprob=-0.0004204819560982287, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:606', bytes=[110, 97, 109, 101], logprob=-3.1709168979432434e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:788', bytes=[34, 58], logprob=-8.344646857949556e-07, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:330', bytes=[32, 34], logprob=-0.00013124081306159496, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:59394', bytes=[100, 105, 118, 105, 100, 101], logprob=-0.12195511907339096, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:497', bytes=[34, 44], logprob=-6.318072337307967e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:330', bytes=[32, 34], logprob=-9.65590606938349e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:16370', bytes=[97, 114, 103, 117, 109, 101, 110, 116, 115], logprob=-0.00040141629870049655, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:788', bytes=[34, 58], logprob=-8.701899787411094e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:5212', bytes=[32, 123, 34], logprob=-0.004756564274430275, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:64', bytes=[97], logprob=-0.0001110968878492713, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:788', bytes=[34, 58], logprob=-5.960462772236497e-07, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-0.0037193186581134796, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:19', bytes=[52], logprob=-0.0041105784475803375, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-0.0002366024418734014, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-3.862306402879767e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:11', bytes=[44], logprob=-0.005275377072393894, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:330', bytes=[32, 34], logprob=-1.7881377516459906e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:65', bytes=[98], logprob=-0.0003137096355203539, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:788', bytes=[34, 58], logprob=-2.264974000354414e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-0.00023910524032544345, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:17', bytes=[50], logprob=-0.6938489675521851, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-8.868777513271198e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:11248', bytes=[125, 125, 10], logprob=-0.014010997489094734, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:151658', bytes=[60, 47, 116, 111, 111, 108, 95, 99, 97, 108, 108, 62], logprob=-0.0029475123155862093, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:151645', bytes=[], logprob=-0.0002113357331836596, top_logprobs=[])], refusal=None), message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='chatcmpl-tool-a8c24beba5ac4bdf8bd97bc31630e30d', function=Function(arguments='{"a": 400, "b": 20}', name='divide'), type='function')], reasoning_content=None), stop_reason=None)
{'role': 'tool', 'tool_call_id': 'chatcmpl-tool-a8c24beba5ac4bdf8bd97bc31630e30d', 'content': '20.0'}
Choice(finish_reason='stop', index=0, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='token_id:785', bytes=[84, 104, 101], logprob=-0.03439528867602348, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:1102', bytes=[32, 114, 101, 115, 117, 108, 116], logprob=-0.12837809324264526, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:315', bytes=[32, 111, 102], logprob=-0.0065645999275147915, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-0.38478943705558777, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:19', bytes=[52], logprob=-2.5987286790041253e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-1.1920922133867862e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-1.1920928244535389e-07, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:608', bytes=[32, 47], logprob=-0.9812964200973511, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:320', bytes=[32, 40], logprob=-0.00030083899036981165, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:16', bytes=[49], logprob=-2.1457441107486375e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-1.9192511899746023e-05, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:488', bytes=[32, 43], logprob=-0.0003904534096363932, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-4.887569048150908e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:16', bytes=[49], logprob=-1.1920922133867862e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-1.1920928244535389e-07, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:8', bytes=[41], logprob=-0.00013422065239865333, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:374', bytes=[32, 105, 115], logprob=-0.0008557948167435825, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:220', bytes=[32], logprob=-0.0009078433504328132, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:17', bytes=[50], logprob=-0.0002949994814116508, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-3.933898824470816e-06, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:13', bytes=[46], logprob=-0.0018945855554193258, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:15', bytes=[48], logprob=-0.0008897398365661502, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:13', bytes=[46], logprob=-0.004208399448543787, top_logprobs=[]), ChatCompletionTokenLogprob(token='token_id:151645', bytes=[], logprob=-0.01276665460318327, top_logprobs=[])], refusal=None), message=ChatCompletionMessage(content='The result of 400 / (10 + 10) is 20.0.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content=None), stop_reason=None)
模型输出: The result of 400 / (10 + 10) is 20.0.
[rank0]:[W819 20:45:44.636916587 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())


# 训练模型的保存目录
.art/mcp_alphavantage/models/mcp-14b-alpha-001/
# tree /workspace/verl/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/
├── checkpoints
│   ├── 0001
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0002
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0003
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0004
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0005
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0006
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── 0007
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   └── 0008
│       ├── README.md
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── added_tokens.json
│       ├── chat_template.jinja
│       ├── merges.txt
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── vocab.json
├── history.jsonl
├── logs
│   └── vllm.log
├── model.json
├── tensors
│   ├── advantages.pt
│   ├── assistant_mask.pt
│   ├── group_ids.pt
│   ├── input_pos.pt
│   ├── logprobs.pt
│   ├── parent_ids.pt
│   ├── tokens.pt
│   └── weights.pt
└── trajectories
    ├── train
    │   ├── 0000.jsonl
    │   ├── 0001.jsonl
    │   ├── 0002.jsonl
    │   ├── 0003.jsonl
    │   ├── 0004.jsonl
    │   ├── 0005.jsonl
    │   ├── 0006.jsonl
    │   └── 0007.jsonl
    └── val
        ├── 0000.jsonl
        ├── 0001.jsonl
        ├── 0002.jsonl
        ├── 0003.jsonl
        ├── 0004.jsonl
        ├── 0005.jsonl
        ├── 0006.jsonl
        └── 0007.jsonl