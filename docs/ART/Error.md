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
输出错误日志
[openai_server_task] 启动 OpenAI 兼容 server 协程...
[openai_server_task] 使用 base_url=http://0.0.0.0:8000/v1, api_key=default
[openai_server_task] 等待 server 启动，超时时间: 200.0 秒
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
[openai_server_task.test_client] server 未就绪，重试中... 错误: Error code: 502
[openai_server_task.test_client] server 未就绪，重试中... 错误: Error code: 502
[openai_server_task.test_client] server 未就绪，重试中... 错误: Error code: 502
[openai_server_task.test_client] server 未就绪，重试中... 错误: Error code: 502
[openai_server_task.test_client] server 未就绪，重试中... 错误: Error code: 502
[openai_server_task.test_client] server 未就绪，重试中... 错误: Error code: 502

pip freeze | grep openai
openai==1.99.1

根据官方vllm issue应该是取消，http的proxy环境变量才行，否则都会报错502
https://github.com/vllm-project/vllm/issues/1519


# 训练报错
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
在move.py的_handle_request函数中加上详细的日志后，出现的报错如下，发现是huggingface的问题
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/workspace/verl/backend/ART_mcp-rl/mcp_rl/train.py", line 246, in <module>
    main()
  File "/workspace/verl/backend/ART_mcp-rl/mcp_rl/train.py", line 236, in main
    asyncio.run(train_mcp_agent(model, use_skypilot=args.use_skypilot))
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 201, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/usr/lib/python3.10/asyncio/tasks.py", line 234, in __step
    result = coro.throw(exc)
  File "/workspace/verl/backend/ART_mcp-rl/mcp_rl/train.py", line 116, in train_mcp_agent
    await model.register(backend)
  File "/workspace/verl/ART/src/art/model.py", line 310, in register
    base_url, api_key = await backend._prepare_backend_for_training(
  File "/workspace/verl/ART/src/art/local/backend.py", line 269, in _prepare_backend_for_training
    await service.start_openai_server(config=config)
  File "/workspace/verl/ART/src/mp_actors/traceback.py", line 26, in async_wrapper
    raise e.with_traceback(streamlined_traceback())
  File "/usr/lib/python3.10/asyncio/futures.py", line 285, in __await__
    yield self  # This tells Task to wait for completion.
  File "/usr/lib/python3.10/asyncio/tasks.py", line 304, in __wakeup
    future.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 201, in result
    raise self._exception.with_traceback(self._exception_tb)
RuntimeError: Unpicklable exception: ConnectionError(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /api/models/unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7446e1683e20>: Failed to establish a new connection: [Errno 101] Network is unreachable'))"), '(Request ID: ecfdd9c7-7787-4009-8994-254fa9185698)')


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

# 卡住， 然后kill掉, 或者一直按ctrl+c结束
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


# weave初始化失败， 环境变量里面加： export WANDB_MODE=offline
raceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/__init__.py", line 4, in <module>
    from .rollout import McpScenario, rollout
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/rollout.py", line 30, in <module>
    weave.init("mcp-agent-training")
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/api.py", line 98, in init
    initialized_client = weave_init.init_weave(
                         ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_init.py", line 120, in init_weave
    entity_name, project_name = get_entity_project_from_project_name(project_name)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_init.py", line 54, in get_entity_project_from_project_name
    entity_name = api.default_entity_name()
                  ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/weave/wandb_interface/wandb_api.py", line 175, in default_entity_name
    result = self.query(self.VIEWER_DEFAULT_ENTITY_QUERY)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/weave/wandb_interface/wandb_api.py", line 158, in query
    return session.execute(query, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/gql/client.py", line 1016, in execute
    result = self._execute(
             ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/gql/client.py", line 925, in _execute
    result = self.transport.execute(
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/gql/transport/requests.py", line 237, in execute
    response = self.session.request(
               ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/requests/adapters.py", line 694, in send
    raise ProxyError(e, request=request)
requests.exceptions.ProxyError: HTTPSConnectionPool(host='api.wandb.ai', port=443): Max retries exceeded with url: /graphql (Caused by ProxyError('Unable to connect to proxy', NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f42003a7200>: Failed to establish a new connection: [Errno 111] Connection refused')))

# weave和wandb不一致时
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_init.py", line 122, in init_weave
    weave_client.check_wandb_run_matches(wandb_run_id, entity_name, project_name)
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_client.py", line 2391, in check_wandb_run_matches
    raise ValueError(
ValueError: Project Mismatch: weave and wandb must be initialized using the same project. Found wandb.init targeting project "johnson/mcp_alphavantage" and weave.init targeting project "johnson-/mcp_alphavantage". To fix, please use the same project for both library initializations.
强制设置，让它们一致
if os.getenv("WANDB_API_KEY"):
    print("Initializing Weave 和 Wandb")
    wandb.init(
        project="mcp_alphavantage",
        entity="johnson-"
    )
    weave.init("mcp_alphavantage")
  

# wandb和 weave  #rm wandb 然后尽量取消wandb和wave的使用
wandb: WARNING `resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id mcp-14b-alpha-001.
wandb: Tracking run with wandb version 0.21.0
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Training failed with error: Project Mismatch: weave and wandb must be initialized using the same project. Found wandb.init targeting project "/mcp_alphavantage" and weave.init targeting project "johnson-/mcp_alphavantage". To fix, please use the same project for both library initializations.
Traceback (most recent call last):
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 226, in main
    asyncio.run(train_mcp_agent(model, use_skypilot=args.use_skypilot))
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
           ^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/futures.py", line 203, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/usr/lib/python3.12/asyncio/tasks.py", line 314, in __step_run_and_handle_result
    result = coro.send(None)
             ^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 109, in train_mcp_agent
    await model.register(backend)
  File "/workspace/verl/RLDecisionAgent/ART/src/art/model.py", line 308, in register
    await super().register(backend)
  File "/workspace/verl/RLDecisionAgent/ART/src/art/model.py", line 146, in register
    await self._backend.register(self)
  File "/workspace/verl/RLDecisionAgent/ART/src/art/local/backend.py", line 122, in register
    _ = self._get_wandb_run(model)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/local/backend.py", line 511, in _get_wandb_run
    self._weave_clients[model.name] = weave.init(model.project)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/api.py", line 98, in init
    initialized_client = weave_init.init_weave(
                         ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_init.py", line 122, in init_weave
    weave_client.check_wandb_run_matches(wandb_run_id, entity_name, project_name)
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_client.py", line 2391, in check_wandb_run_matches
    raise ValueError(
ValueError: Project Mismatch: weave and wandb must be initialized using the same project. Found wandb.init targeting project "/mcp_alphavantage" and weave.init targeting project "johnson-/mcp_alphavantage". To fix, please use the same project for both library initializations.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 236, in <module>
    main()
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 226, in main
    asyncio.run(train_mcp_agent(model, use_skypilot=args.use_skypilot))
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
           ^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/futures.py", line 203, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/usr/lib/python3.12/asyncio/tasks.py", line 314, in __step_run_and_handle_result
    result = coro.send(None)
             ^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 109, in train_mcp_agent
    await model.register(backend)
  File "/workspace/verl/RLDecisionAgent/ART/src/art/model.py", line 308, in register
    await super().register(backend)
  File "/workspace/verl/RLDecisionAgent/ART/src/art/model.py", line 146, in register
    await self._backend.register(self)
  File "/workspace/verl/RLDecisionAgent/ART/src/art/local/backend.py", line 122, in register
    _ = self._get_wandb_run(model)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/local/backend.py", line 511, in _get_wandb_run
    self._weave_clients[model.name] = weave.init(model.project)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/api.py", line 98, in init
    initialized_client = weave_init.init_weave(
                         ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_init.py", line 122, in init_weave
    weave_client.check_wandb_run_matches(wandb_run_id, entity_name, project_name)
  File "/usr/local/lib/python3.12/dist-packages/weave/trace/weave_client.py", line 2391, in check_wandb_run_matches
    raise ValueError(
ValueError: Project Mismatch: weave and wandb must be initialized using the same project. Found wandb.init targeting project "/mcp_alphavantage" and weave.init targeting project "johnson-/mcp_alphavantage". To fix, please use the same project for both library initializations.


# 训练时CUDA错误， 检查nvidia-smi是否正常，如果不正常，请重启容器，检查宿主机的GPU状态
Training failed with error: CUDA error: operation not permitted
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Traceback (most recent call last):
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 235, in main
    asyncio.run(train_mcp_agent(model, use_skypilot=args.use_skypilot))
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
           ^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/futures.py", line 203, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/usr/lib/python3.12/asyncio/tasks.py", line 316, in __step_run_and_handle_result
    result = coro.throw(exc)
             ^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 116, in train_mcp_agent
    await model.register(backend)
  File "/workspace/verl/RLDecisionAgent/ART/src/art/model.py", line 310, in register
    base_url, api_key = await backend._prepare_backend_for_training(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/local/backend.py", line 269, in _prepare_backend_for_training
    await service.start_openai_server(config=config)
  File "/workspace/verl/RLDecisionAgent/ART/src/mp_actors/traceback.py", line 26, in async_wrapper
    raise e.with_traceback(streamlined_traceback())
  File "/workspace/verl/RLDecisionAgent/ART/src/art/unsloth/service.py", line 65, in start_openai_server
    self.state.trainer.save_model(lora_path)
    ^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/functools.py", line 995, in __get__
    val = self.func(instance)
    ^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/unsloth/service.py", line 41, in state
    return ModelState(self.config)
    ^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/unsloth/state.py", line 80, in __init__
    unsloth.FastLanguageModel.from_pretrained(**config.get("init_args", {})),
    ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/unsloth/models/loader.py", line 143, in from_pretrained
    patch_unsloth_smart_gradient_checkpointing(dtype = dtype)
    ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/gradient_checkpointing.py", line 778, in patch_unsloth_smart_gradient_checkpointing
    initialize_unsloth_gradient_checkpointing(dtype)
    ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/gradient_checkpointing.py", line 342, in initialize_unsloth_gradient_checkpointing
    EXTRA_STREAMS = tuple([torch.cuda.Stream() if DEVICE_TYPE == "cuda" else torch.xpu.Stream() for i in range(n_gpus)])
      ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/cuda/streams.py", line 37, in __new__
    return super().__new__(cls, priority=priority, **kwargs)
    ^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: operation not permitted
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 245, in <module>
    main()
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 235, in main
    asyncio.run(train_mcp_agent(model, use_skypilot=args.use_skypilot))
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
           ^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/futures.py", line 203, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/usr/lib/python3.12/asyncio/tasks.py", line 314, in __step_run_and_handle_result
    result = coro.send(None)
             ^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/backend/ART_mcp-rl/mcp_rl/train.py", line 116, in train_mcp_agent
    await model.register(backend)
  File "/workspace/verl/RLDecisionAgent/ART/src/art/model.py", line 310, in register
    base_url, api_key = await backend._prepare_backend_for_training(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/local/backend.py", line 267, in _prepare_backend_for_training
    service = await self._get_service(model)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/local/backend.py", line 131, in _get_service
    config = get_model_config(
             ^^^^^^^^^^^^^^^^^
  File "/workspace/verl/RLDecisionAgent/ART/src/art/dev/get_model_config.py", line 47, in get_model_config
    and torch.cuda.get_device_capability()[0] >= 8
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py", line 560, in get_device_capability
    prop = get_device_properties(device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py", line 576, in get_device_properties
    _lazy_init()  # will define _get_device_properties
    ^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py", line 372, in _lazy_init
    torch._C._cuda_init()
RuntimeError: No CUDA GPUs are available

# 训练卡住了，只能在ctrl +c结束， 是ruler模型问题，需要检查网络是否能正常连同openai，或者使用LLM_cache.py文件
[DEBUG] Metrics recorded for future #6
[GatherContext.update_pbar] Called with n=1
[GatherContext.update_pbar] Incrementing pbar by 1
                                                                                                                                    [GatherContext.update_pbar] Metric reward: sum=0.0, divisor=26, avg=0.0ask_completed=0, success=0, ran_out_of_turns=0, llm_completio
[GatherContext.update_pbar] Metric task_completed: sum=0, divisor=26, avg=0.0
[GatherContext.update_pbar] Metric success: sum=0, divisor=26, avg=0.0
[GatherContext.update_pbar] Metric ran_out_of_turns: sum=0, divisor=26, avg=0.0
[GatherContext.update_pbar] Metric llm_completion_duration: sum=37.21663122100108, divisor=26, avg=1.431408893115426
[GatherContext.update_pbar] Metric num_turns: sum=52, divisor=26, avg=2.0
[GatherContext.update_pbar] Metric duration: sum=111.58859999999999, divisor=26, avg=4.2918692307692305
[GatherContext.update_pbar] Metric completion_tokens: sum=770.5, divisor=26, avg=29.634615384615383
[GatherContext.update_pbar] Setting postfix: {'reward': 0.0, 'task_completed': 0.0, 'success': 0.0, 'ran_out_of_turns': 0.0, 'llm_completion_duration': 1.431408893115426, 'num_turns': 2.0, 'duration': 4.2918692307692305, 'completion_tokens': 29.634615384615383}
                                                                                                                                    [DEBUG] Progress bar updated (success), total trajectories=7reward=0, task_completed=0, success=0, ran_out_of_turns=0, llm_completio
[DEBUG] Gather loop finished, total trajectories=7, total exceptions=0
在这里我按了ctrl + C                                                                                                                                    ^C[ERROR] Exception in future #5: CancelledError()2.55it/s, reward=0, task_completed=0, success=0, ran_out_of_turns=0, llm_completio
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 92, in run_until_complete
    self._run_once()
  File "/usr/local/lib/python3.12/dist-packages/nest_asyncio.py", line 115, in _run_once
    event_list = self._selector.select(timeout)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/selectors.py", line 468, in select
    fd_event_list = self._selector.poll(timeout, max_ev)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspace/verl/RLDecisionAgent/ART/src/art/trajectories.py", line 199, in _
    trajectory = await future
                 ^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/tasks.py", line 627, in _wait_for_one
    f = await done.get()
        ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/queues.py", line 158, in get
    await getter
  File "/usr/lib/python3.12/asyncio/futures.py", line 287, in __await__
    yield self  # This tells Task to wait for completion.
    ^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/tasks.py", line 385, in __wakeup
    future.result()
  File "/usr/lib/python3.12/asyncio/futures.py", line 198, in result
    raise exc
asyncio.exceptions.CancelledError

[GatherContext.update_pbar] Called with n=0
[GatherContext.update_pbar] Incrementing pbar by 0
[GatherContext.update_pbar] Metric reward: sum=0.0, divisor=26, avg=0.0
[GatherContext.update_pbar] Metric task_completed: sum=0, divisor=26, avg=0.0
[GatherContext.update_pbar] Metric success: sum=0, divisor=26, avg=0.0
[GatherContext.update_pbar] Metric ran_out_of_turns: sum=0, divisor=26, avg=0.0
[GatherContext.update_pbar] Metric llm_completion_duration: sum=37.21663122100108, divisor=26, avg=1.431408893115426
[GatherContext.update_pbar] Metric num_turns: sum=52, divisor=26, avg=2.0
[GatherContext.update_pbar] Metric duration: sum=111.58859999999999, divisor=26, avg=4.2918692307692305
[GatherContext.update_pbar] Metric completion_tokens: sum=770.5, divisor=26, avg=29.634615384615383
[GatherContext.update_pbar] Metric exceptions: sum=1, divisor=1, avg=1.0
[GatherContext.update_pbar] Setting postfix: {'reward': 0.0, 'task_completed': 0.0, 'success': 0.0, 'ran_out_of_turns': 0.0, 'llm_completion_duration': 1.431408893115426, 'num_turns': 2.0, 'duration': 4.2918692307692305, 'exceptions': 1.0, 'completion_tokens': 29.634615384615383}
                                                                                                                                    [DEBUG] Progress bar updated (exception), total exceptions=1reward=0, task_completed=0, success=0, ran_out_of_turns=0, llm_completio
[GatherContext.too_many_exceptions] exceptions=1, max_exceptions=0
[GatherContext.too_many_exceptions] -> True


# litellm 计算cost
[RULER-DEBUG] Parsed 2 scores from model response.
[RULER-DEBUG] Trajectory updated with score=0.3, explanation=Trajectory 1 makes an attempt using tool calls but fails due to input
validation errors and never computes the expression. It shows some effort but does not provide a correct or complete result.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Trajectory 2 contains no actions or attempts to complete the problem,
and therefore does not contribute toward achieving the task.
[RULER-DEBUG] Returning new TrajectoryGroup with updated scores.
[DEBUG] TrajectoryGroup.__init__ started
[DEBUG] TrajectoryGroup.__init__ finished
[DEBUG] TrajectoryGroup.__init__ started
[DEBUG] TrajectoryGroup.__init__ finished
组 2, 轨迹 2 的奖励差异: 0.3
19:49:45 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: openai/o3-mini-2025-01-31
[2025-08-19 19:49:45] INFO cost_calculator.py:655: selected model name for cost calculation: openai/o3-mini-2025-01-31

# 训练代码卡住， 训练已经结束，只是 dataset 迭代器还没完全释放（或者日志还在显示进度条），看起来像“卡住”了。
DEBUG put request: train () {}████████| 2/2 [00:03<00:00,  1.41s/it, loss=0.0899, grad_norm=1.08, policy_loss=0.0899, entropy=0.252]
[HANDLE_REQUEST] id=b15c632c-f3df-470f-bad7-ba2bc782f29b, method=train, args=(), kwargs={}
[ERROR] Exception in train: <class 'StopAsyncIteration'>
Traceback (most recent call last):
  File "/workspace/verl/ART/src/mp_actors/move.py", line 255, in _handle_request
    result = await generators[request.id].asend(request.send_value)
StopAsyncIteration
[HANDLE_REQUEST_DONE] id=b15c632c-f3df-470f-bad7-ba2bc782f29b, result_type=<class 'NoneType'>, exception_type=<class 'StopAsyncIteration'>
train: 100%|██████████████████████████| 2/2 [00:07<00:00,  3.70s/it, loss=0.0899, grad_norm=1.08, policy_loss=0.0899, entropy=0.252]
[INFO] 步 7 模型训练完成
Iterating dataset: 100%|██████████████████████████████████████████████████████████████████████████| 8/8 [13:29<00:00, 115.59s/batch]


# python -m ART.src.art.openai_patch  测试Openai报错
/usr/lib/python3.10/runpy.py:126: RuntimeWarning: 'ART.src.art.openai_patch' found in sys.modules after import of package 'ART.src.art', but prior to execution of 'ART.src.art.openai_patch'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
Testing patched client.chat.completions.create ...
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/workspace/verl/ART/src/art/openai_patch.py", line 201, in <module>
    asyncio.run(main())
  File "/usr/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/workspace/verl/ART/src/art/openai_patch.py", line 198, in main
    full_completion = await consume_chat_completion_stream(resp, on_chunk)
  File "/workspace/verl/ART/src/art/openai_patch.py", line 172, in consume_chat_completion_stream
    assert chat_completion is not None, f"模型回复的内容为空，请检查"
AssertionError: 模型回复的内容为空，请检查


# 报错
uant_storage': 'uint8', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': ['lm_head', 'multi_modal_projector', 'merger', 'modality_projection', 'model.layers.0.self_attn', 'model.layers.0.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.21.mlp', 'model.layers.0.self_attn.q_proj'], 'llm_int8_threshold': 6.0}
[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/vllm_utils.py", line 1500, in load_vllm
[rank0]:     llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
[rank0]:   File "/workspace/verl/ART/src/art/unsloth/state.py", line 74, in _from_engine_args
[rank0]:     return from_engine_args(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 684, in from_engine_args
[rank0]:     return async_engine_cls.from_vllm_config(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 657, in from_vllm_config
[rank0]:     return cls(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 612, in __init__
[rank0]:     self.engine = self._engine_class(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 267, in __init__
[rank0]:     super().__init__(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 275, in __init__
[rank0]:     self.model_executor = executor_class(vllm_config=vllm_config)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/executor_base.py", line 52, in __init__
[rank0]:     self._init_executor()
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py", line 47, in _init_executor
[rank0]:     self.collective_rpc("load_model")
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py", line 56, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/utils.py", line 2456, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/vllm/worker/worker.py", line 195, in load_model
[rank0]:     assert allocator.get_current_usage() == 0, (
[rank0]: AssertionError: Sleep mode can only be used for one instance per process.

[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/backend/ART_Search_Email/train_email_search_agent.py", line 640, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/verl/backend/ART_Search_Email/train_email_search_agent.py", line 628, in main
[rank0]:     asyncio.run(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 30, in run
[rank0]:     return loop.run_until_complete(task)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 98, in run_until_complete
[rank0]:     return f.result()
[rank0]:   File "/usr/lib/python3.10/asyncio/futures.py", line 201, in result
[rank0]:     raise self._exception.with_traceback(self._exception_tb)
[rank0]:   File "/usr/lib/python3.10/asyncio/tasks.py", line 232, in __step
[rank0]:     result = coro.send(None)
[rank0]:   File "/workspace/verl/backend/ART_Search_Email/train_email_search_agent.py", line 584, in run_training
[rank0]:     await model.train(
[rank0]:   File "/workspace/verl/ART/src/art/model.py", line 356, in train
[rank0]:     async for _ in self.backend()._train_model(
[rank0]:   File "/workspace/verl/ART/src/art/local/backend.py", line 562, in _train_model
[rank0]:     async for result in service.train(
[rank0]:   File "/workspace/verl/ART/src/art/unsloth/service.py", line 123, in train
[rank0]:     trainer=self.state.trainer,
[rank0]:   File "/usr/lib/python3.10/functools.py", line 981, in __get__
[rank0]:     val = self.func(instance)
[rank0]:   File "/workspace/verl/ART/src/art/unsloth/service.py", line 45, in state
[rank0]:     return ModelState(self.config)
[rank0]:   File "/workspace/verl/ART/src/art/unsloth/state.py", line 82, in __init__
[rank0]:     unsloth.FastLanguageModel.from_pretrained(**config.get("init_args", {})),
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth/models/loader.py", line 402, in from_pretrained
[rank0]:     model, tokenizer = dispatch_model.from_pretrained(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth/models/qwen2.py", line 87, in from_pretrained
[rank0]:     return FastLlamaModel.from_pretrained(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 2041, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/vllm_utils.py", line 1527, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError: Sleep mode can only be used for one instance per process.
[rank0]:[W821 20:01:54.081613048 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())


# 检查Openai的网络，或者LLM_cache.py报错了
  File "/workspace/verl/backend/ART_PPT_content/train_test_model/generate_benchmarks.py", line 64, in score_group
    scored_group = await ruler_score_group(
  File "/workspace/verl/ART/src/art/rewards/ruler.py", line 256, in ruler_score_group
    scores = await ruler(
  File "/workspace/verl/ART/src/art/rewards/ruler.py", line 152, in ruler
    response = await acompletion(
  File "/usr/local/lib/python3.10/dist-packages/litellm/utils.py", line 1492, in wrapper_async
    raise e
  File "/usr/local/lib/python3.10/dist-packages/litellm/utils.py", line 1353, in wrapper_async
    result = await original_function(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/litellm/main.py", line 531, in acompletion
    raise exception_type(
  File "/usr/local/lib/python3.10/dist-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 2239, in exception_type
    raise e
  File "/usr/local/lib/python3.10/dist-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 474, in exception_type
    raise APIError(
litellm.exceptions.APIError: litellm.APIError: APIError: OpenAIException - 'str' object has no attribute 'model_dump'


# 在使用模型前必须先注册，例如await gpt_4o_mini.register(backend)
Traceback (most recent call last):
  File "/workspace/verl/backend/ART_PPT_content/train_test_model/generate_benchmarks.py", line 242, in <module>
    asyncio.run(run_benchmarks(args.server))
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 201, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/usr/lib/python3.10/asyncio/tasks.py", line 232, in __step
    result = coro.send(None)
  File "/workspace/verl/backend/ART_PPT_content/train_test_model/generate_benchmarks.py", line 226, in run_benchmarks
    await log_comparison_model(comparison_model, val_scenarios, control_groups)
  File "/workspace/verl/backend/ART_PPT_content/train_test_model/generate_benchmarks.py", line 117, in log_comparison_model
    await comparison_model.log(
  File "/workspace/verl/ART/src/art/model.py", line 224, in log
    await self.backend()._log(
  File "/workspace/verl/ART/src/art/model.py", line 131, in backend
    raise ValueError(
ValueError: Model is not registered with the Backend. You must call `model.register()` first.


# 报错 model.train需要使用TrajectoryGroup的形成的列表
        for group in train_groups:
            valid_trajectories = [traj for traj in group if isinstance(traj, art.Trajectory)]
            if len(valid_trajectories) > 1:
                valid_train_groups.append(art.TrajectoryGroup(valid_trajectories))

[gather_trajectory_groups] Finished.
Traceback (most recent call last):
  File "/workspace/verl/backend/ART_title_generator/train.py", line 339, in <module>
    asyncio.run(main())
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 201, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/usr/lib/python3.10/asyncio/tasks.py", line 232, in __step
    result = coro.send(None)
  File "/workspace/verl/backend/ART_title_generator/train.py", line 319, in main
    await model.train(valid_train_groups, config=art.TrainConfig(learning_rate=LEARNING_RATE))
  File "/workspace/verl/ART/src/art/model.py", line 356, in train
    async for _ in self.backend()._train_model(
  File "/workspace/verl/ART/src/]/local/backend.py", line 422, in _train_model
    await self._log(model, trajectory_groups, "train")
  File "/workspace/verl/ART/src/art/local/backend.py", line 352, in _log
    f.write(serialize_trajectory_groups(trajectory_groups))
  File "/workspace/verl/ART/src/art/utils/trajectory_logging.py", line 13, in serialize_trajectory_groups
    group_dicts = [
  File "/workspace/verl/ART/src/art/utils/trajectory_logging.py", line 14, in <listcomp>
    trajectory_group_to_dict(trajectory_group)
  File "/workspace/verl/ART/src/art/utils/trajectory_logging.py", line 23, in trajectory_group_to_dict
    for trajectory in trajectory_group.trajectories:
AttributeError: 'list' object has no attribute 'trajectories'


#加载模型时，没有指定lora配置
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
        # 关键：为保证加载正确，测试时需提供与训练时相同的内部配置
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(gpu_memory_utilization=0.75),
            peft_args=art.dev.PeftArgs(lora_alpha=8),
            trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
        ),
    )

Unsloth 2025.8.6 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers.
[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/backend/ART_title_generator/model_test.py", line 108, in <module>
[rank0]:     asyncio.run(main())
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 30, in run
[rank0]:     return loop.run_until_complete(task)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 98, in run_until_complete
[rank0]:     return f.result()
[rank0]:   File "/usr/lib/python3.10/asyncio/futures.py", line 201, in result
[rank0]:     raise self._exception.with_traceback(self._exception_tb)
[rank0]:   File "/usr/lib/python3.10/asyncio/tasks.py", line 232, in __step
[rank0]:     result = coro.send(None)
[rank0]:   File "/workspace/verl/backend/ART_title_generator/model_test.py", line 45, in main
[rank0]:     await model.register(backend)
[rank0]:   File "/workspace/verl/ART/src/art/model.py", line 310, in register
[rank0]:     base_url, api_key = await backend._prepare_backend_for_training(
[rank0]:   File "/workspace/verl/ART/src/art/local/backend.py", line 322, in _prepare_backend_for_training
[rank0]:     await service.start_openai_server(config=config)
[rank0]:   File "/workspace/verl/ART/src/art/unsloth/service.py", line 87, in start_openai_server
[rank0]:     engine=self.state.vllm.async_engine,
[rank0]:   File "/usr/lib/python3.10/functools.py", line 981, in __get__
[rank0]:     val = self.func(instance)
[rank0]:   File "/workspace/verl/ART/src/art/unsloth/service.py", line 41, in state
[rank0]:     return ModelState(self.config)
[rank0]:   File "/workspace/verl/ART/src/art/unsloth/state.py", line 89, in __init__
[rank0]:     unsloth.FastLanguageModel.get_peft_model(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 2383, in get_peft_model
[rank0]:     raise TypeError(
[rank0]: TypeError: Unsloth: Your model already has LoRA adapters. Your new parameters are different.
[rank0]:[W823 09:29:26.531051531 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

# 报错, 正常的结束的报错
[2025-08-27 14:39:08] ERROR base_events.py:1758: Exception in callback LocalBackend._prepare_backend_for_training.<locals>.done_callback(<Task cancell...ckend.py:287>>) at /usr/local/lib/python3.10/dist-packages/art/local/backend.py:278
handle: <Handle LocalBackend._prepare_backend_for_training.<locals>.done_callback(<Task cancell...ckend.py:287>>) at /usr/local/lib/python3.10/dist-packages/art/local/backend.py:278>
Traceback (most recent call last):
  File "/usr/lib/python3.10/asyncio/tasks.py", line 234, in __step
    result = coro.throw(exc)
  File "/usr/local/lib/python3.10/dist-packages/mp_actors/move.py", line 102, in _handle_responses
    response: Response = await loop.run_in_executor(
  File "/usr/lib/python3.10/asyncio/futures.py", line 285, in __await__
    yield self  # This tells Task to wait for completion.
  File "/usr/lib/python3.10/asyncio/tasks.py", line 304, in __wakeup
    future.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 196, in result
    raise exc
asyncio.exceptions.CancelledError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/asyncio/events.py", line 80, in _run
    self._context.run(self._callback, *self._args)
  File "/usr/local/lib/python3.10/dist-packages/art/local/backend.py", line 279, in done_callback
    close_proxy(self._services.pop(model.name))
  File "/usr/local/lib/python3.10/dist-packages/mp_actors/move.py", line 60, in close_proxy
    getattr(proxy, "close", lambda: None)()
  File "/usr/local/lib/python3.10/dist-packages/mp_actors/move.py", line 214, in close
    asyncio.get_event_loop().run_until_complete(self._handle_responses_task)
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 196, in result
    raise exc
asyncio.exceptions.CancelledError


#  报错
LiteLLM completion() model= gpt-4o-mini; provider = openai
[2025-08-27 21:19:45] INFO _client.py:1740: HTTP Request: POST http://127.0.0.1:6688/chat/completions "HTTP/1.1 200 OK"
21:19:47 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: openai/gpt-4o-mini-2024-07-18
[2025-08-27 21:19:47] INFO cost_calculator.py:655: selected model name for cost calculation: openai/gpt-4o-mini-2024-07-18
21:19:47 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: openai/gpt-4o-mini-2024-07-18
[2025-08-27 21:19:47] INFO cost_calculator.py:655: selected model name for cost calculation: openai/gpt-4o-mini-2024-07-18
gather:  25%|▎| 2/8 [00:44<02:14, 22.47s/it, exceptions=6, reward=0.263, structure_score=0.263, structure/slides_count=0.25
[2025-08-27 21:19:47] ERROR base_events.py:1758: Exception in callback LocalBackend._prepare_backend_for_training.<locals>.done_callback(<Task cancell...ckend.py:287>>) at /usr/local/lib/python3.10/dist-packages/art/local/backend.py:278
handle: <Handle LocalBackend._prepare_backend_for_training.<locals>.done_callback(<Task cancell...ckend.py:287>>) at /usr/local/lib/python3.10/dist-packages/art/local/backend.py:278>
Traceback (most recent call last):
  File "/usr/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/workspace/verl/backend/ART_Langgraph_multi/train.py", line 637, in main
    jg = await ruler_score_group(g, RULER_MODEL, extra_litellm_params=extra_litellm_params, debug=True)
  File "/usr/local/lib/python3.10/dist-packages/art/rewards/ruler.py", line 232, in ruler_score_group
    raise ValueError("Additional histories are not supported by RULER yet.")
ValueError: Additional histories are not supported by RULER yet.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/asyncio/events.py", line 80, in _run
    self._context.run(self._callback, *self._args)
  File "/usr/local/lib/python3.10/dist-packages/art/local/backend.py", line 279, in done_callback
    close_proxy(self._services.pop(model.name))
  File "/usr/local/lib/python3.10/dist-packages/mp_actors/move.py", line 60, in close_proxy
    getattr(proxy, "close", lambda: None)()
  File "/usr/local/lib/python3.10/dist-packages/mp_actors/move.py", line 214, in close
    asyncio.get_event_loop().run_until_complete(self._handle_responses_task)
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 196, in result
    raise exc
asyncio.exceptions.CancelledError
Traceback (most recent call last):
  File "/workspace/verl/backend/ART_Langgraph_multi/train.py", line 664, in <module>
    asyncio.run(main())
  File "/usr/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/workspace/verl/backend/ART_Langgraph_multi/train.py", line 637, in main
    jg = await ruler_score_group(g, RULER_MODEL, extra_litellm_params=extra_litellm_params, debug=True)
  File "/usr/local/lib/python3.10/dist-packages/art/rewards/ruler.py", line 232, in ruler_score_group
    raise ValueError("Additional histories are not supported by RULER yet.")
ValueError: Additional histories are not supported by RULER yet.
Traceback (most recent call last):
  File "/workspace/verl/backend/ART_Langgraph_multi/train.py", line 664, in <module>
    asyncio.run(main())
  File "/usr/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/workspace/verl/backend/ART_Langgraph_multi/train.py", line 637, in main
    jg = await ruler_score_group(g, RULER_MODEL, extra_litellm_params=extra_litellm_params, debug=True)
  File "/usr/local/lib/python3.10/dist-packages/art/rewards/ruler.py", line 232, in ruler_score_group
    raise ValueError("Additional histories are not supported by RULER yet.")
ValueError: Additional histories are not supported by RULER yet.


# 训练时报错，看exceptions数量增加，说明rollout出现问题
Unsloth: Just some info: will skip parsing ['k_norm', 'q_norm', 'post_feedforward_layernorm', 'pre_feedforward_layernorm']
Unsloth: Just some info: will skip parsing ['k_norm', 'q_norm', 'post_feedforward_layernorm', 'pre_feedforward_layernorm']
Unsloth 2025.8.6 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers.
/home/wac/johnson/.pycharm_helpers/pydev/pydevd_plugins/__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  __import__('pkg_resources').declare_namespace(__name__)
Iterating dataset:   0%|                               | 0/2 [00:00<?, ?batch/s][train] step=0 epoch=0
开始训练场景：1
开始训练场景：2
gather:   0%|                                             | 0/8 [00:00<?, ?it/s]Rollout scenario step 0
Rollout scenario step 0
Rollout scenario step 0
Rollout scenario step 0
Rollout scenario step 0
Rollout scenario step 0
Rollout scenario step 0
Rollout scenario step 0
gather:   0%|                                             | 0/8 [00:06<?, ?it/s]
gather:   0%|                               | 0/8 [00:06<?, ?it/s, exceptions=1]
gather:   0%|                               | 0/8 [00:06<?, ?it/s, exceptions=2]
gather:   0%|                               | 0/8 [00:06<?, ?it/s, exceptions=3]
gather:   0%|                               | 0/8 [00:06<?, ?it/s, exceptions=3]
gather:   0%|                               | 0/8 [00:06<?, ?it/s, exceptions=4]
gather:   0%|                               | 0/8 [00:06<?, ?it/s, exceptions=4]
gather:   0%|                               | 0/8 [00:06<?, ?it/s, exceptions=5]
gather:   0%|                               | 0/8 [00:06<?, ?it/s, exceptions=6]
gather:   0%|                               | 0/8 [00:07<?, ?it/s, exceptions=6]
gather:   0%|                               | 0/8 [00:07<?, ?it/s, exceptions=7]
gather:   0%|                               | 0/8 [00:07<?, ?it/s, exceptions=8]
Skipping tuning as there is no suitable data. This can happen when all the trajectories in the same group have the same reward and thus no advantage to train on.
Advanced step from 0 to 1 (no training occurred)
Iterating dataset:  50%|███████████▌           | 1/2 [00:08<00:08,  8.81s/batch][train] step=1 epoch=1
开始训练场景：2
开始训练场景：1
gather:   0%|                                             | 0/8 [00:00<?, ?it/s]Rollout scenario step 1
Rollout scenario step 1
Rollout scenario step 1
Rollout scenario step 1
Rollout scenario step 1
Rollout scenario step 1
Rollout scenario step 1
Rollout scenario step 1
gather:   0%|                                             | 0/8 [00:04<?, ?it/s]
gather:   0%|                               | 0/8 [00:04<?, ?it/s, exceptions=1]
gather:   0%|                               | 0/8 [00:04<?, ?it/s, exceptions=1]
gather:   0%|                               | 0/8 [00:04<?, ?it/s, exceptions=2]
gather:   0%|                               | 0/8 [00:05<?, ?it/s, exceptions=2]
gather:   0%|                               | 0/8 [00:05<?, ?it/s, exceptions=3]
gather:   0%|                               | 0/8 [00:05<?, ?it/s, exceptions=3]
gather:   0%|                               | 0/8 [00:05<?, ?it/s, exceptions=4]
gather:   0%|                               | 0/8 [00:05<?, ?it/s, exceptions=5]
gather:   0%|                               | 0/8 [00:05<?, ?it/s, exceptions=6]
gather:   0%|                               | 0/8 [00:05<?, ?it/s, exceptions=7]
gather:   0%|                               | 0/8 [00:08<?, ?it/s, exceptions=7]
gather:   0%|                               | 0/8 [00:08<?, ?it/s, exceptions=8]
Skipping tuning as there is no suitable data. This can happen when all the trajectories in the same group have the same reward and thus no advantage to train on.
Advanced step from 1 to 2 (no training occurred)
Iterating dataset: 100%|███████████████████████| 2/2 [00:17<00:00,  8.89s/batch]

# 加载模型和运行模型不一致，请修改NAME字段，确保它们是唯一的，否则会加载已有的模型（即以前运行过的配置)
import sys; print('Python %s on %s' % (sys.version, sys.platform)) /home/wac/johnson/anaconda3/bin/conda run -n gpt --no-capture-output python /home/wac/johnson/.pycharm_helpers/pydev/pydevd.py --multiprocess --qt-support=auto --client localhost --port 35877 --file /media/wac/backup/john/johnson/RLDecisionAgent/backend/ART_Langgraph/model_test.py /home/wac/johnson/.pycharm_helpers/pydev/pydevd_plugins/__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. 
The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81. __import__('pkg_resources').declare_namespace(__name__) Connected to pydev debugger (build 241.18968.29) 测试项目名：web-search-agent-training, 模型名：Qwen/Qwen2.5-7B-Instruct, Name: web-search wandb: Currently logged in as: johnson to http://192.168.100.8:3005. Use wandb login --relogin to force relogin wandb: Tracking run with wandb version 0.21.0 wandb: Run data is saved locally in /media/wac/backup/john/johnson/RLDecisionAgent/backend/ART_Langgraph/wandb/run-20250829_130637-web-search wandb: Run wandb offline to turn off syncing. wandb: Resuming run web-search wandb: ⭐️ View project at http://192.168.100.8:3005/johnson/web-search-agent-training wandb: 🚀 View run at http://192.168.100.8:3005/johnson/web-search-agent-training/runs/web-search /home/wac/johnson/.pycharm_helpers/pydev/pydevd_plugins/__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81. __import__('pkg_resources').declare_namespace(__name__) INFO 08-29 13:07:00 [importing.py:53] Triton module has been replaced with a placeholder. INFO 08-29 13:07:01 [__init__.py:239] Automatically detected platform cuda. pkill: killing pid 1371838 failed: 不允许的操作 pkill: killing pid 1374550 failed: 不允许的操作 pkill: killing pid 1375701 failed: 不允许的操作 pkill: killing pid 1380895 failed: 不允许的操作 pkill: killing pid 1382211 failed: 不允许的操作 pkill: killing pid 1664622 failed: 不允许的操作 pkill: killing pid 1689565 failed: 不允许的操作 pkill: killing pid 1919148 failed: 不允许的操作 pkill: killing pid 1924921 failed: 不允许的操作 pkill: killing pid 1926362 failed: 不允许的操作 pkill: killing pid 1927119 failed: 不允许的操作 pkill: killing pid 1928272 failed: 不允许的操作 pkill: killing pid 1932568 failed: 不允许的操作 pkill: killing pid 1933265 failed: 不允许的操作 pkill: killing pid 1941779 failed: 不允许的操作 pkill: killing pid 1987523 failed: 不允许的操作 pkill: killing pid 2019284 failed: 不允许的操作 pkill: killing pid 2020696 failed: 不允许的操作 pkill: killing pid 2022043 failed: 不允许的操作 pkill: killing pid 2026203 failed: 不允许的操作 pkill: killing pid 2027261 failed: 不允许的操作 pkill: killing pid 2030732 failed: 不允许的操作 pkill: killing pid 2032536 failed: 不允许的操作 /home/wac/johnson/.pycharm_helpers/pydev/pydevd_plugins/__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81. __import__('pkg_resources').declare_namespace(__name__) /home/wac/johnson/.pycharm_helpers/pydev/pydevd_plugins/__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81. __import__('pkg_resources').declare_namespace(__name__) /media/wac/backup/john/johnson/RLDecisionAgent/ART/src/art/__init__.py:10: UserWarning: WARNING: Unsloth should be imported before transformers, peft to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations. Please restructure your imports with 'import unsloth' at the top of your file. import unsloth # type: ignore # noqa: F401 🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning. INFO 08-29 13:07:30 [importing.py:53] Triton module has been replaced with a placeholder. INFO 08-29 13:07:30 [__init__.py:239] Automatically detected platform cuda. 🦥 Unsloth Zoo will now patch everything to make training faster! 测试项目名：web-search-agent-training, 模型名：Qwen/Qwen2.5-7B-Instruct, Name: web-search Unsloth: Patching vLLM v1 graph capture Unsloth: Patching vLLM v0 graph capture ==((====))== Unsloth 2025.8.6: Fast Qwen2 patching. Transformers: 4.55.2. vLLM: 0.8.5.post1. \\ /| NVIDIA GeForce RTX 4090 D. Num GPUs = 1. Max memory: 23.546 GB. Platform: Linux. O^O/ \_/ \ Torch: 2.6.0+cu124. CUDA: 8.9. CUDA Toolkit: 12.4. Triton: 3.2.0 \ / Bfloat16 = TRUE. FA [Xformers = 0.0.29.post2. FA2 = True] "-____-" Free license: http://github.com/unslothai/unsloth Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored! Unsloth: vLLM loading unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit with actual GPU utilization = 77.65% Unsloth: Your GPU has CUDA compute capability 8.9 with VRAM = 23.55 GB. Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 32768. Num Sequences = 288. Unsloth: vLLM's KV Cache can use up to 17.81 GB. Also swap space = 6 GB. INFO 08-29 13:08:21 [config.py:717] This model supports multiple tasks: {'generate', 'classify', 'score', 'reward', 'embed'}. Defaulting to 'generate'. Unsloth: vLLM Bitsandbytes config using kwargs = {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_quant_storage': 'uint8', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': ['lm_head', 'multi_modal_projector', 'merger', 'modality_projection', 'model.layers.0.self_attn', 'model.layers.0.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.21.mlp', 'model.layers.0.self_attn.q_proj'], 'llm_int8_threshold': 6.0} INFO 08-29 13:08:21 [llm_engine.py:240] Initializing a V0 LLM engine (v0.8.5.post1) with config: model='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', speculative_config=None, tokenizer='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=bitsandbytes, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='auto', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit, num_scheduler_steps=16, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"level":0,"backend":"inductor","splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"epilogue_fusion":true,"max_autotune":false,"shape_padding":true,"trace.enabled":false,"triton.cudagraphs":true,"debug":false,"dce":true,"memory_planning":true,"coordinate_descent_tuning":true,"trace.graph_diagram":false,"compile_threads":32,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.multi_kernel":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,"triton.cooperative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchmark_combo_kernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":288}, use_cached_outputs=False, INFO 08-29 13:08:25 [cuda.py:292] Using Flash Attention backend. INFO 08-29 13:08:25 [parallel_state.py:1004] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0 INFO 08-29 13:08:25 [model_runner.py:1108] Starting to load model unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit... INFO 08-29 13:08:25 [loader.py:1187] Loading weights with BitsAndBytes quantization. May take a while ... INFO 08-29 13:08:28 [weight_utils.py:265] Using model weights format ['*.safetensors'] INFO 08-29 13:08:29 [weight_utils.py:281] Time spent downloading weights for unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit: 1.145630 seconds INFO 08-29 13:08:30 [weight_utils.py:315] No model.safetensors.index.json found in remote. Loading safetensors checkpoint shards: 0% Completed | 0/1 [00:00<?, ?it/s] Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 2.30it/s] Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 2.30it/s] Loading safetensors checkpoint shards: 0% Completed | 0/1 [00:00<?, ?it/s] Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 2.11it/s] Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 2.11it/s] INFO 08-29 13:08:31 [punica_selector.py:18] Using PunicaWrapperGPU. INFO 08-29 13:08:32 [model_runner.py:1140] Model loading took 0.5153 GiB and 6.204724 seconds INFO 08-29 13:08:35 [worker.py:287] Memory profiling takes 2.72 seconds INFO 08-29 13:08:35 [worker.py:287] the current vLLM instance can use total_gpu_memory (23.55GiB) x gpu_memory_utilization (0.78) = 18.28GiB INFO 08-29 13:08:35 [worker.py:287] model weights take 0.52GiB; non_torch_memory takes 0.07GiB; PyTorch activation peak memory takes 1.61GiB; the rest of the memory reserved for KV Cache is 16.09GiB. INFO 08-29 13:08:35 [executor_base.py:112] # cuda blocks: 87853, # CPU blocks: 32768 INFO 08-29 13:08:35 [executor_base.py:117] Maximum concurrency for 32768 tokens per request: 42.90x INFO 08-29 13:08:39 [vllm_utils.py:671] Unsloth: Running patched vLLM v0 capture_model. INFO 08-29 13:08:39 [model_runner.py:1450] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing gpu_memory_utilization or switching to eager mode. You can also reduce the max_num_seqs as needed to decrease memory usage. Capturing CUDA graph shapes: 100%|██████████████| 39/39 [00:22<00:00, 1.70it/s] INFO 08-29 13:09:02 [model_runner.py:1592] Graph capturing finished in 23 secs, took 0.55 GiB INFO 08-29 13:09:02 [vllm_utils.py:678] Unsloth: Patched vLLM v0 graph capture finished in 23 secs. INFO 08-29 13:09:03 [llm_engine.py:437] init engine (profile, create kv cache, warmup model) took 30.99 seconds Unsloth: Just some info: will skip parsing ['q_norm', 'k_norm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm'] Unsloth: Just some info: will skip parsing ['q_norm', 'k_norm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm'] Unsloth 2025.8.6 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers. Unsloth: Already have LoRA adapters! We shall skip this step. 测试的输出结果: {'messages': [SystemMessage(content='\nYou are a web search agent. Use tools to find information on the web.\nWhen done, provide a concise answer.\n', additional_kwargs={}, response_metadata={}, id='0c93e94c-6578-4ef8-8d6a-4a82cf37fbcb'), HumanMessage(content='Who is the CFO of Tesla?', additional_kwargs={}, response_metadata={}, id='fa6d0e79-137b-44db-af96-0cdc829edab8'), AIMessage(content="To provide you with the most accurate response, I would recommend you to search for specific information. I can check the most recent filings of Tesla or search for the latest press releases for more up-to-date facts on the company's situation.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 178, 'total_tokens': 226, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'web-search', 'system_fingerprint': None, 'id': 'chatcmpl-f552daf13c884b6fafdc3d4db8df6c3a', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': {'content': [{'token': 'token_id:1249', 'bytes': [84, 111], 'logprob': -1.5664010047912598, 'top_logprobs': []}, {'token': 'token_id:3410', 'bytes': [32, 112, 114, 111, 118, 105, 100, 101], 'logprob': -0.20715616643428802, 'top_logprobs': []}, {'token': 'token_id:498', 'bytes': [32, 121, 111, 117], 'logprob': -0.38067907094955444, 'top_logprobs': []}, {'token': 'token_id:448', 'bytes': [32, 119, 105, 116, 104], 'logprob': -0.000169382052263245, 'top_logprobs': []}, {'token': 'token_id:279', 'bytes': [32, 116, 104, 101], 'logprob': -0.4950447678565979, 'top_logprobs': []}, {'token': 'token_id:1429', 'bytes': [32, 109, 111, 115, 116], 'logprob': -0.07142695784568787, 'top_logprobs': []}, {'token': 'token_id:13382', 'bytes': [32, 97, 99, 99, 117, 114, 97, 116, 101], 'logprob': -0.07657865434885025, 'top_logprobs': []}, {'token': 'token_id:2033', 'bytes': [32, 114, 101, 115, 112, 111, 110, 115, 101], 'logprob': -4.804951190948486, 'top_logprobs': []}, {'token': 'token_id:11', 'bytes': [44], 'logprob': -0.03034316562116146, 'top_logprobs': []}, {'token': 'token_id:358', 'bytes': [32, 73], 'logprob': -0.42374667525291443, 'top_logprobs': []}, {'token': 'token_id:1035', 'bytes': [32, 119, 111, 117, 108, 100], 'logprob': -0.8945327997207642, 'top_logprobs': []}, {'token': 'token_id:6934', 'bytes': [32, 114, 101, 99, 111, 109, 109, 101, 110, 100], 'logprob': -1.6517952680587769, 'top_logprobs': []}, {'token': 'token_id:498', 'bytes': [32, 121, 111, 117], 'logprob': -2.6865665912628174, 'top_logprobs': []}, {'token': 'token_id:311', 'bytes': [32, 116, 111], 'logprob': -0.16849109530448914, 'top_logprobs': []}, {'token': 'token_id:2711', 'bytes': [32, 115, 101, 97, 114, 99, 104], 'logprob': -1.4202699661254883, 'top_logprobs': []}, {'token': 'token_id:369', 'bytes': [32, 102, 111, 114], 'logprob': -0.6889943480491638, 'top_logprobs': []}, {'token': 'token_id:3151', 'bytes': [32, 115, 112, 101, 99, 105, 102, 105, 99], 'logprob': -6.388165473937988, 'top_logprobs': []}, {'token': 'token_id:1995', 'bytes': [32, 105, 110, 102, 111, 114, 109, 97, 116, 105, 111, 110], 'logprob': -0.591657280921936, 'top_logprobs': []}, {'token': 'token_id:13', 'bytes': [46], 'logprob': -3.452016830444336, 'top_logprobs': []}, {'token': 'token_id:358', 'bytes': [32, 73], 'logprob': -2.8039393424987793, 'top_logprobs': []}, {'token': 'token_id:646', 'bytes': [32, 99, 97, 110], 'logprob': -0.4983573853969574, 'top_logprobs': []}, {'token': 'token_id:1779', 'bytes': [32, 99, 104, 101, 99, 107], 'logprob': -5.2719407081604, 'top_logprobs': []}, {'token': 'token_id:279', 'bytes': [32, 116, 104, 101], 'logprob': -1.089690089225769, 'top_logprobs': []}, {'token': 'token_id:1429', 'bytes': [32, 109, 111, 115, 116], 'logprob': -2.8323187828063965, 'top_logprobs': []}, {'token': 'token_id:3213', 'bytes': [32, 114, 101, 99, 101, 110, 116], 'logprob': -0.4139953851699829, 'top_logprobs': []}, {'token': 'token_id:67148', 'bytes': [32, 102, 105, 108, 105, 110, 103, 115], 'logprob': -6.299983501434326, 'top_logprobs': []}, {'token': 'token_id:315', 'bytes': [32, 111, 102], 'logprob': -1.4139783382415771, 'top_logprobs': []}, {'token': 'token_id:27199', 'bytes': [32, 84, 101, 115, 108, 97], 'logprob': -0.6949511170387268, 'top_logprobs': []}, {'token': 'token_id:476', 'bytes': [32, 111, 114], 'logprob': -2.472810745239258, 'top_logprobs': []}, {'token': 'token_id:2711', 'bytes': [32, 115, 101, 97, 114, 99, 104], 'logprob': -2.811878204345703, 'top_logprobs': []}, {'token': 'token_id:369', 'bytes': [32, 102, 111, 114], 'logprob': -0.7450668215751648, 'top_logprobs': []}, {'token': 'token_id:279', 'bytes': [32, 116, 104, 101], 'logprob': -1.4107160568237305, 'top_logprobs': []}, {'token': 'token_id:5535', 'bytes': [32, 108, 97, 116, 101, 115, 116], 'logprob': -1.7625718116760254, 'top_logprobs': []}, {'token': 'token_id:3493', 'bytes': [32, 112, 114, 101, 115, 115], 'logprob': -3.589028835296631, 'top_logprobs': []}, {'token': 'token_id:19232', 'bytes': [32, 114, 101, 108, 101, 97, 115, 101, 115], 'logprob': -0.5332621335983276, 'top_logprobs': []}, {'token': 'token_id:369', 'bytes': [32, 102, 111, 114], 'logprob': -1.9575753211975098, 'top_logprobs': []}, {'token': 'token_id:803', 'bytes': [32, 109, 111, 114, 101], 'logprob': -1.9221185445785522, 'top_logprobs': []}, {'token': 'token_id:705', 'bytes': [32, 117, 112], 'logprob': -2.072411060333252, 'top_logprobs': []}, {'token': 'token_id:4686', 'bytes': [45, 116, 111], 'logprob': -0.018792560324072838, 'top_logprobs': []}, {'token': 'token_id:18413', 'bytes': [45, 100, 97, 116, 101], 'logprob': -0.003404774935916066, 'top_logprobs': []}, {'token': 'token_id:13064', 'bytes': [32, 102, 97, 99, 116, 115], 'logprob': -4.90785026550293, 'top_logprobs': []}, {'token': 'token_id:389', 'bytes': [32, 111, 110], 'logprob': -2.095146417617798, 'top_logprobs': []}, {'token': 'token_id:279', 'bytes': [32, 116, 104, 101], 'logprob': -0.23652587831020355, 'top_logprobs': []}, {'token': 'token_id:2813', 'bytes': [32, 99, 111, 109, 112, 97, 110, 121], 'logprob': -2.3276422023773193, 'top_logprobs': []}, {'token': 'token_id:594', 'bytes': [39, 115], 'logprob': -0.1682809740304947, 'top_logprobs': []}, {'token': 'token_id:6534', 'bytes': [32, 115, 105, 116, 117, 97, 116, 105, 111, 110], 'logprob': -7.4729323387146, 'top_logprobs': []}, {'token': 'token_id:13', 'bytes': [46], 'logprob': -0.1643821746110916, 'top_logprobs': []}, {'token': 'token_id:151645', 'bytes': [], 'logprob': -0.4224308729171753, 'top_logprobs': []}], 'refusal': None}}, id='run--afd41a1d-146b-4136-bf07-7f0b1abfd180-0', usage_metadata={'input_tokens': 178, 'output_tokens': 48, 'total_tokens': 226, 'input_token_details': {}, 'output_token_details': {}})]} [TEST] agent finished. See backend logs / tracing for details.

# 不训练错误，直接结束，大部分是继续训练时，num_epochs数量和已经训练的数量相当，就没有在继续训练了，直接结束了，
可以修改训练的name或者增大numm_epochs数量
wandb: WARNING `start_method` is deprecated and will be removed in a future version of wandb. This setting is currently non-functional and safely ignored.
wandb: Currently logged in as: johnson to http://192.168.100.8:3005. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.21.0
wandb: Run data is saved locally in /media/wac/backup/john/johnson/RLDecisionAgent/backend/ART_Langgraph/wandb/run-20250829_211336-8gft27rm
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run web-search-20250829-211336
wandb: ⭐️ View project at http://192.168.100.8:3005/johnson/web-search-agent-training
wandb: 🚀 View run at http://192.168.100.8:3005/johnson/web-search-agent-training/runs/8gft27rm
wandb: wandb.init() called while a run is active and reinit is set to 'default', so returning the previous run.
INFO 08-29 21:13:52 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 08-29 21:13:52 [__init__.py:239] Automatically detected platform cuda.
pkill: killing pid 1371838 failed: 不允许的操作
pkill: killing pid 1374550 failed: 不允许的操作
pkill: killing pid 1375701 failed: 不允许的操作
pkill: killing pid 1380895 failed: 不允许的操作
pkill: killing pid 1382211 failed: 不允许的操作
pkill: killing pid 1664622 failed: 不允许的操作
pkill: killing pid 1689565 failed: 不允许的操作
pkill: killing pid 1919148 failed: 不允许的操作
pkill: killing pid 1924921 failed: 不允许的操作
pkill: killing pid 1926362 failed: 不允许的操作
pkill: killing pid 1927119 failed: 不允许的操作
pkill: killing pid 1928272 failed: 不允许的操作
pkill: killing pid 1932568 failed: 不允许的操作
pkill: killing pid 1933265 failed: 不允许的操作
pkill: killing pid 1941779 failed: 不允许的操作
pkill: killing pid 1987523 failed: 不允许的操作
pkill: killing pid 2019284 failed: 不允许的操作
pkill: killing pid 2020696 failed: 不允许的操作
pkill: killing pid 2022043 failed: 不允许的操作
pkill: killing pid 2026203 failed: 不允许的操作
pkill: killing pid 2027261 failed: 不允许的操作
pkill: killing pid 2030732 failed: 不允许的操作
pkill: killing pid 2032536 failed: 不允许的操作
/media/wac/backup/john/johnson/RLDecisionAgent/ART/src/art/__init__.py:10: UserWarning: WARNING: Unsloth should be imported before transformers, peft to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  import unsloth  # type: ignore # noqa: F401
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
INFO 08-29 21:14:02 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 08-29 21:14:02 [__init__.py:239] Automatically detected platform cuda.
🦥 Unsloth Zoo will now patch everything to make training faster!
Unsloth: Patching vLLM v1 graph capture
Unsloth: Patching vLLM v0 graph capture
==((====))==  Unsloth 2025.8.6: Fast Qwen2 patching. Transformers: 4.55.2. vLLM: 0.8.5.post1.
   \\   /|    NVIDIA GeForce RTX 4090 D. Num GPUs = 1. Max memory: 23.546 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.6.0+cu124. CUDA: 8.9. CUDA Toolkit: 12.4. Triton: 3.2.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post2. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: vLLM loading unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit with actual GPU utilization = 77.65%
Unsloth: Your GPU has CUDA compute capability 8.9 with VRAM = 23.55 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 32768. Num Sequences = 288.
Unsloth: vLLM's KV Cache can use up to 17.81 GB. Also swap space = 6 GB.
INFO 08-29 21:14:36 [config.py:717] This model supports multiple tasks: {'reward', 'generate', 'embed', 'classify', 'score'}. Defaulting to 'generate'.
Unsloth: vLLM Bitsandbytes config using kwargs = {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_quant_storage': 'uint8', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'llm_int8_enable_fp32_cpu_offload': False, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': ['lm_head', 'multi_modal_projector', 'merger', 'modality_projection', 'model.layers.0.self_attn', 'model.layers.0.mlp', 'model.layers.2.mlp', 'model.layers.3.mlp', 'model.layers.21.mlp', 'model.layers.0.self_attn.q_proj'], 'llm_int8_threshold': 6.0}
INFO 08-29 21:14:37 [llm_engine.py:240] Initializing a V0 LLM engine (v0.8.5.post1) with config: model='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', speculative_config=None, tokenizer='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=bitsandbytes, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='auto', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit, num_scheduler_steps=16, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"level":0,"backend":"inductor","splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"epilogue_fusion":true,"max_autotune":false,"shape_padding":true,"trace.enabled":false,"triton.cudagraphs":true,"debug":false,"dce":true,"memory_planning":true,"coordinate_descent_tuning":true,"trace.graph_diagram":false,"compile_threads":32,"group_fusion":true,"disable_progress":false,"verbose_progress":true,"triton.multi_kernel":0,"triton.use_block_ptr":true,"triton.enable_persistent_tma_matmul":true,"triton.autotune_at_compile_time":false,"triton.cooperative_reductions":false,"cuda.compile_opt_level":"-O2","cuda.enable_cuda_lto":true,"combo_kernels":false,"benchmark_combo_kernel":true,"combo_kernel_foreach_dynamic_shapes":true,"enable_auto_functionalized_v2":false},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":288}, use_cached_outputs=False, 
INFO 08-29 21:14:42 [cuda.py:292] Using Flash Attention backend.
INFO 08-29 21:14:42 [parallel_state.py:1004] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0
INFO 08-29 21:14:42 [model_runner.py:1108] Starting to load model unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit...
INFO 08-29 21:14:43 [loader.py:1187] Loading weights with BitsAndBytes quantization. May take a while ...
INFO 08-29 21:14:46 [weight_utils.py:265] Using model weights format ['*.safetensors']
INFO 08-29 21:14:47 [weight_utils.py:281] Time spent downloading weights for unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit: 1.031381 seconds
INFO 08-29 21:14:49 [weight_utils.py:315] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.99it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.98it/s]

Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.74it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.74it/s]

INFO 08-29 21:14:50 [punica_selector.py:18] Using PunicaWrapperGPU.
INFO 08-29 21:14:50 [model_runner.py:1140] Model loading took 0.5153 GiB and 7.174856 seconds
INFO 08-29 21:14:53 [worker.py:287] Memory profiling takes 2.72 seconds
INFO 08-29 21:14:53 [worker.py:287] the current vLLM instance can use total_gpu_memory (23.55GiB) x gpu_memory_utilization (0.78) = 18.28GiB
INFO 08-29 21:14:53 [worker.py:287] model weights take 0.52GiB; non_torch_memory takes 0.07GiB; PyTorch activation peak memory takes 1.61GiB; the rest of the memory reserved for KV Cache is 16.09GiB.
INFO 08-29 21:14:54 [executor_base.py:112] # cuda blocks: 87853, # CPU blocks: 32768
INFO 08-29 21:14:54 [executor_base.py:117] Maximum concurrency for 32768 tokens per request: 42.90x
INFO 08-29 21:14:57 [vllm_utils.py:671] Unsloth: Running patched vLLM v0 `capture_model`.
INFO 08-29 21:14:57 [model_runner.py:1450] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|██████████████| 39/39 [00:13<00:00,  2.98it/s]
INFO 08-29 21:15:10 [model_runner.py:1592] Graph capturing finished in 13 secs, took 0.55 GiB
INFO 08-29 21:15:10 [vllm_utils.py:678] Unsloth: Patched vLLM v0 graph capture finished in 13 secs.
INFO 08-29 21:15:11 [llm_engine.py:437] init engine (profile, create kv cache, warmup model) took 21.04 seconds
Unsloth: Just some info: will skip parsing ['k_norm', 'q_norm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm']
Unsloth: Just some info: will skip parsing ['k_norm', 'q_norm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm']
Unsloth 2025.8.6 patched 24 layers with 24 QKV layers, 24 O layers and 24 MLP layers.
Unsloth: Already have LoRA adapters! We shall skip this step.
Iterating dataset: 6batch [00:00, ?batch/s]
wandb:                                                                                
wandb: 
wandb: Run history:
wandb: train/step ▁
wandb: 
wandb: Run summary:
wandb: train/step 0
wandb: 
wandb: 🚀 View run web-search-20250829-211336 at: http://192.168.100.8:3005/johnson/web-search-agent-training/runs/8gft27rm
wandb: ⭐️ View project at: http://192.168.100.8:3005/johnson/web-search-agent-training
wandb: Synced 6 W&B file(s), 1 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250829_211336-8gft27rm/logs


# 迭代到一定进度后意外停止， tqdm 默认 leave=True，被打断时会把最后一次进度留在控制台，看起来就像“停在 12%”。其实是训练完成了
wandb: 🚀 View run web-search03-20250901-144200 at: http://192.168.100.8:3005/johnson/web-search-agent-training/runs/aimkca2f
wandb: ⭐️ View project at: http://192.168.100.8:3005/johnson/web-search-agent-training
wandb: Synced 6 W&B file(s), 2 media file(s), 4 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250901_144201-aimkca2f/logs
wandb: WARNING Tried to log to step 7 that is less than the current step 8. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
Iterating dataset:  12%|██████████▌                                                                                | 7/60 [00:43<?, ?batch/s]
[2025-09-01 14:44:49] ERROR base_events.py:1758: Exception in callback LocalBackend._prepare_backend_for_training.<locals>.done_callback(<Task cancell...ckend.py:287>>) at /usr/local/lib/python3.10/dist-packages/art/local/backend.py:278
handle: <Handle LocalBackend._prepare_backend_for_training.<locals>.done_callback(<Task cancell...ckend.py:287>>) at /usr/local/lib/python3.10/dist-packages/art/local/backend.py:278>
Traceback (most recent call last):
  File "/usr/lib/python3.10/asyncio/tasks.py", line 234, in __step
    result = coro.throw(exc)
  File "/usr/local/lib/python3.10/dist-packages/mp_actors/move.py", line 102, in _handle_responses
    response: Response = await loop.run_in_executor(
  File "/usr/lib/python3.10/asyncio/futures.py", line 285, in __await__
    yield self  # This tells Task to wait for completion.
  File "/usr/lib/python3.10/asyncio/tasks.py", line 304, in __wakeup
    future.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 196, in result
    raise exc
asyncio.exceptions.CancelledError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/asyncio/events.py", line 80, in _run
    self._context.run(self._callback, *self._args)
  File "/usr/local/lib/python3.10/dist-packages/art/local/backend.py", line 279, in done_callback
    close_proxy(self._services.pop(model.name))
  File "/usr/local/lib/python3.10/dist-packages/mp_actors/move.py", line 60, in close_proxy
    getattr(proxy, "close", lambda: None)()
  File "/usr/local/lib/python3.10/dist-packages/mp_actors/move.py", line 214, in close
    asyncio.get_event_loop().run_until_complete(self._handle_responses_task)
  File "/usr/local/lib/python3.10/dist-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
  File "/usr/lib/python3.10/asyncio/futures.py", line 196, in result
    raise exc
asyncio.exceptions.CancelledError


# 注意reward为0时不会进行训练
2025-09-03 13:19:00,236 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
                                                                                                                          2025-09-03 13:19:03,349 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"2]
                                                                                                                          2025-09-03 13:19:19,883 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"8]
                                                                                                                          2025-09-03 13:20:13,070 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"5]
2025-09-03 13:20:14,243 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
gather: 100%|██████████████████████████████████████████████| 6/6 [01:56<00:00, 19.43s/it, reward=0, completion_tokens=861]
Skipping tuning as there is no suitable data. This can happen when all the trajectories in the same group have the same reward and thus no advantage to train on.
Advanced step from 4 to 5 (no training occurred)
Iterating dataset:  50%|███████████████████████████████▌                               | 5/10 [10:04<09:59, 119.81s/batch][train] step=5 epoch=5
                                                                                                                          2025-09-03 13:20:22,710 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"s]
                                                                                                                          2025-09-03 13:20:23,113 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"0]
                                                                                                                          2025-09-03 13:21:12,569 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"6]
2025-09-03 13:21:15,248 [INFO] httpx:1025 - HTTP Request: POST https://open.bigmodel.cn/api/paas/v4/web_search "HTTP/1.1 200 OK"
2025-09-03 13:21:23,061 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
                                                                                                                          2025-09-03 13:22:35,758 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"4]
                                                                                                                          2025-09-03 13:23:32,361 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"5]
2025-09-03 13:23:37,823 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
                                                                                                                          2025-09-03 13:25:10,651 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"8]
2025-09-03 13:26:52,312 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-03 13:27:02,755 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/completions "HTTP/1.1 200 OK"



2025-09-03 13:28:52,323 [INFO] httpx:1740 - HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
gather: 100%|█| 6/6 [08:37<00:00, 86.33s/it, reward=0.113, format_reward=0.588, search_reward=0.77, sources_count=8, compl
Packed 3 trajectories into 2 sequences of length 30720.113, format_reward=0.588, search_reward=0.77, sources_count=8, comp
                                                                                                                          ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1                               | 0/2 [00:00<?, ?it/s]
   \\   /|    Num examples = 10,000,000 | Num Epochs = 3 | Total steps = 30,000,000
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 1 x 1) = 2
 "-____-"     Trainable parameters = 20,185,088 of 7,635,801,600 (0.26% trained)
Unsloth: Will smartly offload gradients to save VRAM!
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 10,000,000 | Num Epochs = 3 | Total steps = 60,000,000
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 1 x 1) = 1
 "-____-"     Trainable parameters = 20,185,088 of 7,635,801,600 (0.26% trained)

# kill 掉model-service，这个是启动的vllm模型服务，模拟Openai接口
Traceback (most recent call last):
  File "/home/wac/johnson/.pycharm_helpers/pydevd_asyncio/pydevd_nest_asyncio.py", line 138, in run
    return loop.run_until_complete(task)
  File "/home/wac/johnson/.pycharm_helpers/pydevd_asyncio/pydevd_nest_asyncio.py", line 243, in run_until_complete
    return f.result()
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/asyncio/futures.py", line 201, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/asyncio/tasks.py", line 232, in __step
    result = coro.send(None)
  File "/media/wac/backup/john/johnson/RLDecisionAgent/backend/ART_Langgraph/train.py", line 296, in main
    await model.register(backend)
  File "/media/wac/backup/john/johnson/RLDecisionAgent/ART/src/art/model.py", line 322, in register
    base_url, api_key = await backend._prepare_backend_for_training(
  File "/media/wac/backup/john/johnson/RLDecisionAgent/ART/src/art/local/backend.py", line 272, in _prepare_backend_for_training
    await service.start_openai_server(config=config)
  File "/media/wac/backup/john/johnson/RLDecisionAgent/ART/src/mp_actors/traceback.py", line 26, in async_wrapper
    raise e.with_traceback(streamlined_traceback())
  File "/media/wac/backup/john/johnson/RLDecisionAgent/ART/src/art/unsloth/service.py", line 62, in start_openai_server
    self._openai_server_task = await openai_server_task(
  File "/media/wac/backup/john/johnson/RLDecisionAgent/ART/src/art/vllm/server.py", line 103, in openai_server_task
    task.result()
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/asyncio/futures.py", line 201, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/asyncio/tasks.py", line 232, in __step
    result = coro.send(None)
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/vllm/entrypoints/openai/api_server.py", line 1066, in run_server
    sock = create_server_socket(sock_addr)
  File "/home/wac/johnson/anaconda3/envs/gpt/lib/python3.10/site-packages/vllm/entrypoints/openai/api_server.py", line 1037, in create_server_socket
    sock.bind(addr)
OSError: [Errno 98] 地址已在使用


# 训练卡住，如果在下面的步骤中卡住，那么需要使用export CUDA_VISIBLE_DEVICES=1指定显卡，并确保有显存, 不是这个问题，是src/art/unsloth/train.py的中的函数train的trainer.train()卡住
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
