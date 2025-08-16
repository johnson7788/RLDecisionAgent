# 报错，huggingface的模型下载，重试
 File "/workspace/verl/ART/examples/mcp-rl/mcp_rl/train.py", line 107, in train_mcp_agent
    await model.register(backend)
  File "/workspace/verl/ART/src/art/model.py", line 308, in register
    base_url, api_key = await backend._prepare_backend_for_training(
  File "/workspace/verl/ART/src/art/local/backend.py", line 268, in _prepare_backend_for_training
    await service.start_openai_server(config=config)
  File "/workspace/verl/ART/src/mp_actors/traceback.py", line 26, in async_wrapper
    raise e.with_traceback(streamlined_traceback())
  File "/workspace/verl/ART/src/art/unsloth/service.py", line 56, in start_openai_server
    self.state.trainer.save_model(lora_path)
  File "/usr/lib/python3.10/functools.py", line 981, in __get__
    val = self.func(instance)
  File "/workspace/verl/ART/src/art/unsloth/service.py", line 41, in state
    return ModelState(self.config)
  File "/workspace/verl/ART/src/art/unsloth/state.py", line 85, in __init__
    unsloth.FastLanguageModel.from_pretrained(**config.get("init_args", {})),
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/loader.py", line 394, in from_pretrained
    model, tokenizer = dispatch_model.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/qwen2.py", line 87, in from_pretrained
    return FastLlamaModel.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 2027, in from_pretrained
    llm = load_vllm(**load_vllm_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/vllm_utils.py", line 1525, in load_vllm
    raise RuntimeError(error)
RuntimeError: Data processing error: CAS service error : Error : single flight error: Real call failed: ReqwestMiddlewareError(Reqwest(reqwest::Error { kind: Request, url: "https://transfer.xethub.hf.co/xorbs/default/f889b21bc4908258d2f65552c964b2ad449e66e1db5ba54010b79f7ab647c118?X-Xet-Signed-Range=bytes%3D0-57388423&Expires=1754996679&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC9mODg5YjIxYmM0OTA4MjU4ZDJmNjU1NTJjOTY0YjJhZDQ0OWU2NmUxZGI1YmE1NDAxMGI3OWY3YWI2NDdjMTE4P1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTczODg0MjMiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTQ5OTY2Nzl9fX1dfQ__&Signature=ecjMJAhxUpgEtnc6XEGrMXw-n9bwNZLz1yZonxLjQnHr1FF2orqdB3U0xe9dEbT6vowDg5pxHCvquRYdcAng2F2~S7CbtwtB2UHTn0RNVjMYTUmqkaVt~Zsb4R4gopOLgt6ddwNe-UMGgwi7tbcgegXNPwOCODVh6tXmG8m~KbgdOAG~NAp5aZnK8PYj-IVOvhJNNkFmI3SQxdEaO~sTm~oXNNrVaycwgzB7Iekp35UbGqvCy-fVMNgHGZcGOGNWwE0EvQ6ERnY939U823GVoQTlzT9y9qhKD~eYIOpRjnxtmkeBYIfAucpTk~1ICl4cgfAuUbL~Id6ovMwzm9qBRQ__&Key-Pair-Id=K2L8F4GPSG1IFC", source: hyper_util::client::legacy::Error(Connect, Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" }) }))

# 报错，rm -rf wandb
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_init.py", line 122, in init_weave
    weave_client.check_wandb_run_matches(wandb_run_id, entity_name, project_name)
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_client.py", line 2391, in check_wandb_run_matches
    raise ValueError(
ValueError: Project Mismatch: weave and wandb must be initialized using the same project. Found wandb.init targeting project "/mcp_alphavantage" and weave.init targeting project "johnson-/mcp_alphavantage". To fix, please use the same project for both library initializations.

# 报错， 查询VLLM报错原因： find -type f -name vllm.log 2>/dev/null | head -n 5
  File "/workspace/verl/RLDecisionAgent/ART/src/art/local/backend.py", line 268, in _prepare_backend_for_training
    await service.start_openai_server(config=config)
  File "/workspace/verl/RLDecisionAgent/ART/src/mp_actors/traceback.py", line 26, in async_wrapper
    raise e.with_traceback(streamlined_traceback())
  File "/workspace/verl/RLDecisionAgent/ART/src/art/unsloth/service.py", line 58, in start_openai_server
    self._openai_server_task = await openai_server_task(
    ^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/vllm/server.py", line 82, in openai_server_task
    raise TimeoutError(
    ^^^^^^^^^^^^^^^^^
TimeoutError: Unable to reach OpenAI-compatible server within 30.0 seconds. You can increase this timeout by setting the ART_SERVER_TIMEOUT environment variable.

- 查看日志
cat ./ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/logs/vllm.log

# 卡住， 然后kill掉
ps aux | grep mcp_rl.train | grep -v grep | awk '{print $2}' | xargs kill -9


# 报错, 不要安装==0.3.11.post5版本，因为没有after_each
Training failed with error: gather_trajectory_groups() got an unexpected keyword argument 'after_each'
Traceback (most recent call last):
  File "examples/mcp-rl/mcp_rl/train.py", line 224, in main
    asyncio.run(train_mcp_agent(model, use_skypilot=args.use_skypilot))
  File "/home/vipuser/miniconda3/lib/python3.12/site-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vipuser/miniconda3/lib/python3.12/site-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
           ^^^^^^^^^^
  File "/home/vipuser/miniconda3/lib/python3.12/asyncio/futures.py", line 203, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/home/vipuser/miniconda3/lib/python3.12/asyncio/tasks.py", line 314, in __step_run_and_handle_result
    result = coro.send(None)
             ^^^^^^^^^^^^^^^
  File "/root/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 143, in train_mcp_agent
    groups = await art.gather_trajectory_groups(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: gather_trajectory_groups() got an unexpected keyword argument 'after_each'


# 报错 pip install unsloth==2025.8.4 unsloth-zoo==2025.8.3
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/root/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 234, in <module>
    main()
  File "/root/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 224, in main
    asyncio.run(train_mcp_agent(model, use_skypilot=args.use_skypilot))
  File "/home/vipuser/miniconda3/lib/python3.12/site-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vipuser/miniconda3/lib/python3.12/site-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
           ^^^^^^^^^^
  File "/home/vipuser/miniconda3/lib/python3.12/asyncio/futures.py", line 203, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/home/vipuser/miniconda3/lib/python3.12/asyncio/tasks.py", line 316, in __step_run_and_handle_result
    result = coro.throw(exc)
             ^^^^^^^^^^^^^^^
  File "/root/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 107, in train_mcp_agent
    await model.register(backend)
  File "/root/RLDecisionAgent/ART/src/art/model.py", line 308, in register
    base_url, api_key = await backend._prepare_backend_for_training(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/RLDecisionAgent/ART/src/art/local/backend.py", line 268, in _prepare_backend_for_training
    await service.start_openai_server(config=config)
  File "/root/RLDecisionAgent/ART/src/mp_actors/traceback.py", line 26, in async_wrapper
    raise e.with_traceback(streamlined_traceback())
  File "/root/RLDecisionAgent/ART/src/art/unsloth/service.py", line 56, in start_openai_server
    self.state.trainer.save_model(lora_path)
    ^^^^^^^^^^^^^^^^^
  File "/home/vipuser/miniconda3/lib/python3.12/functools.py", line 993, in __get__
    val = self.func(instance)
    ^^^^^^^^^^^^^^^^^
  File "/root/RLDecisionAgent/ART/src/art/unsloth/service.py", line 41, in state
    return ModelState(self.config)
    ^^^^^^^^^^^^^^^^^
  File "/root/RLDecisionAgent/ART/src/art/unsloth/state.py", line 85, in __init__
    unsloth.FastLanguageModel.from_pretrained(**config.get("init_args", {})),
    ^^^^^^^^^^^^^^^^^
  File "/home/vipuser/miniconda3/lib/python3.12/site-packages/unsloth/models/loader.py", line 363, in from_pretrained
    patch_vllm()
  File "/root/RLDecisionAgent/ART/src/art/vllm/patches.py", line 254, in patch_vllm
    vllm_utils.patch_vllm_set_inductor_config()
    ^^^^^^^^^^^^^^^^^
AttributeError: module 'unsloth_zoo.vllm_utils' has no attribute 'patch_vllm_set_inductor_config'

ART/src/art/local/backend.py传入的的config为空
await service.start_openai_server(config=config)
继续调用    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None: 函数
ART/src/art/unsloth/service.py
输入日志:
[start_openai_server] 开始启动 OpenAI server...
[start_openai_server] 当前 output_dir: /workspace/verl/RLDecisionAgent/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001
[start_openai_server] 传入 config: None
[start_openai_server] get_last_checkpoint_dir 返回: None
[start_openai_server] 未找到 checkpoint，尝试使用 step=0 checkpoint
[start_openai_server] step=0 checkpoint 路径: /workspace/verl/RLDecisionAgent/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/checkpoints/0000
[start_openai_server] 已确保目录存在: /workspace/verl/RLDecisionAgent/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/checkpoints
[start_openai_server] 保存初始模型中...
[2025-08-16 22:36:39,241] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-08-16 22:36:39,732] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[start_openai_server] 初始模型保存完成
[start_openai_server] 停止可能已有的 OpenAI server...
[start_openai_server] 旧的 OpenAI server 已停止
[start_openai_server] 准备启动新的 openai_server_task，配置如下：
  - model_name: mcp-14b-alpha-001
  - base_model: Qwen/Qwen2.5-0.5B-Instruct
  - log_file: /workspace/verl/RLDecisionAgent/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/logs/vllm.log
  - lora_path: /workspace/verl/RLDecisionAgent/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/checkpoints/0000
  - config: {'log_file': '/workspace/verl/RLDecisionAgent/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/logs/vllm.log', 'server_args': {'api_key': 'default', 'lora_modules': ['{"name": "mcp-14b-alpha-001", "path": "/workspace/verl/RLDecisionAgent/ART/.art/mcp_alphavantage/models/mcp-14b-alpha-001/checkpoints/0000"}'], 'return_tokens_as_token_ids': True, 'enable_auto_tool_choice': True, 'tool_call_parser': 'hermes'}, 'engine_args': {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'num_scheduler_steps': 16, 'served_model_name': 'Qwen/Qwen2.5-0.5B-Instruct', 'disable_log_requests': True, 'generation_config': 'vllm'}}
发现这里卡住
self._openai_server_task = await openai_server_task(
            engine=self.state.vllm.async_engine,
            config=final_config,
        )
是调用了ART/src/art/vllm/server.py的函数
async def openai_server_task(
    engine: EngineClient,
    config: OpenAIServerConfig,
) -> asyncio.Task[None]:



# 代码报错，需要wandb offline, 不是wandb offline，是有一定几率报错
  import unsloth  # type: ignore # noqa: F401
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
INFO 08-14 22:19:08 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 08-14 22:19:08 [__init__.py:239] Automatically detected platform cuda.
Initializing Weave
Traceback (most recent call last):
  File "/home/vipuser/miniconda3/lib/python3.12/multiprocessing/queues.py", line 264, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vipuser/miniconda3/lib/python3.12/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
TypeError: cannot pickle 'SSLContext' object


# 训练报错，MCP的的请求受到频率限制
Task completion attempted with summary: Attempted to fetch the daily Relative Strength Index (RSI) for Apple (AAPL) but encountered an error ('Technical Analysis: RSI'). Unable to proceed with the analysis due to the unavailability of the required data.
Task completion attempted with summary: Retrieved the daily time series data for Tesla (TSLA). However, the API returned a demo limitation message, indicating the need for a full API key to access the complete data. No further analysis could be performed due to this restriction.
Task completion attempted with summary: Retrieved the daily time series data for Tesla (TSLA). However, the API key provided is for demo purposes only, and the full data could not be accessed. To proceed with a thorough analysis and report, a valid API key is required.
Task completion attempted with summary: Attempted to fetch the daily Relative Strength Index (RSI) for Apple (AAPL) but encountered an error ('Technical Analysis: RSI'). Unable to proceed with the analysis and report due to the error.
Task completion attempted with summary: Retrieved the daily time series data for Tesla (TSLA). However, due to the use of a demo API key, the full data was not accessible. To proceed with a thorough analysis and report, a valid API key is required.
Task completion attempted with summary: The task could not be completed due to API limitations. The demo API key provided does not allow access to the full functionality required to search for stock symbols matching 'Tesla' and analyze the top result. Please upgrade to a full API key to proceed with the task.
Task completion attempted with summary: Attempted to fetch the daily Relative Strength Index (RSI) for Apple (AAPL) but encountered an error: 'Technical Analysis: RSI'. No further analysis or report could be generated due to this issue.
Task completion attempted with summary: Retrieved the daily time series data for Tesla (TSLA). However, the API key used is for demo purposes only, and the full data could not be accessed. To proceed with a thorough analysis and report, a valid API key is required.
Task completion attempted with summary: Attempted to perform a technical analysis for IBM by comparing the 30-day SMA and the 14-day RSI. However, encountered errors while retrieving the SMA and RSI data, indicating potential issues with the data source or API. No further analysis could be conducted.
Task completion attempted with summary: Attempted to perform a technical analysis for IBM by comparing the 30-day SMA and the 14-day RSI. However, the tool encountered errors while retrieving the SMA and RSI data, preventing the completion of the analysis.
Task completion attempted with summary: Attempted to perform a technical analysis for IBM by comparing the 30-day SMA and the 14-day RSI. However, encountered errors while retrieving the SMA and RSI data, preventing the completion of the analysis and report.
gather gpt-4o:   6%| | 2/32 [00:21<04:25,  8.83s/it, reward=0, task_completed=1, success=0, ran_out_of_turns=0, llm_completion_duraTask completion attempted with summary: Attempted to calculate the 60-minute Simple Moving Average (SMA) for IBM and retrieve daily time series data. However, the API returned a demo limitation message, indicating the need for a full API key to access the required data. No further analysis could be performed due to this restriction.