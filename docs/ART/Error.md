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


# 训练报错，需要wandb offline, 不是wandb offline的原因，是有一定几率报错, https://github.com/OpenPipe/ART/issues/343
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

# 训练卡住了，只能在ctrl +c结束
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