# 报错，可以这样， export CUDA_VISIBLE_DEVICES=1， 一定要加export，否则就会报错
[INFO:swift] Setting args.lazy_tokenize: False
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/swift/cli/rlhf.py", line 5, in <module>
    rlhf_main()
  File "/usr/local/lib/python3.11/site-packages/swift/llm/train/rlhf.py", line 200, in rlhf_main
    return SwiftRLHF(args).main()
           ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/swift/llm/train/sft.py", line 27, in __init__
    super().__init__(args)
  File "/usr/local/lib/python3.11/site-packages/swift/llm/base.py", line 19, in __init__
    self.args = self._parse_args(args)
                ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/swift/llm/base.py", line 31, in _parse_args
    args, remaining_argv = parse_args(self.args_class, args)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/swift/utils/utils.py", line 152, in parse_args
    args, remaining_args = parser.parse_args_into_dataclasses(argv, return_remaining_strings=True)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/transformers/hf_argparser.py", line 358, in parse_args_into_dataclasses
    obj = dtype(**inputs)
          ^^^^^^^^^^^^^^^
  File "<string>", line 435, in __init__
  File "/usr/local/lib/python3.11/site-packages/swift/llm/argument/rlhf_args.py", line 135, in __post_init__
    TrainArguments.__post_init__(self)
  File "/usr/local/lib/python3.11/site-packages/swift/llm/argument/train_args.py", line 168, in __post_init__
    self._init_deepspeed()
  File "/usr/local/lib/python3.11/site-packages/swift/llm/argument/train_args.py", line 186, in _init_deepspeed
    raise ValueError('DeepSpeed is not compatible with `device_map`. '
ValueError: DeepSpeed is not compatible with `device_map`. n_gpu: 3, local_world_size: 1.


[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/local/lib/python3.11/site-packages/swift/cli/rlhf.py", line 5, in <module>
[rank0]:     rlhf_main()
[rank0]:   File "/usr/local/lib/python3.11/site-packages/swift/llm/train/rlhf.py", line 200, in rlhf_main
[rank0]:     return SwiftRLHF(args).main()
[rank0]:            ^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/swift/llm/train/sft.py", line 27, in __init__
[rank0]:     super().__init__(args)
[rank0]:   File "/usr/local/lib/python3.11/site-packages/swift/llm/base.py", line 19, in __init__
[rank0]:     self.args = self._parse_args(args)
[rank0]:                 ^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/swift/llm/base.py", line 31, in _parse_args
[rank0]:     args, remaining_argv = parse_args(self.args_class, args)
[rank0]:                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/swift/utils/utils.py", line 152, in parse_args
[rank0]:     args, remaining_args = parser.parse_args_into_dataclasses(argv, return_remaining_strings=True)
[rank0]:                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/transformers/hf_argparser.py", line 358, in parse_args_into_dataclasses
[rank0]:     obj = dtype(**inputs)
[rank0]:           ^^^^^^^^^^^^^^^
[rank0]:   File "<string>", line 435, in __init__
[rank0]:   File "/usr/local/lib/python3.11/site-packages/swift/llm/argument/rlhf_args.py", line 137, in __post_init__
[rank0]:     self._check_grpo()
[rank0]:   File "/usr/local/lib/python3.11/site-packages/swift/llm/argument/rlhf_args.py", line 299, in _check_grpo
[rank0]:     raise ValueError('GRPO with vLLM is not compatible with `device_map`. '
[rank0]: ValueError: GRPO with vLLM is not compatible with `device_map`. Please set NPROC_PER_NODE equal to num_processes.



# 如果在swift rollout中使用multi_turn_scheduler，但是没有指定external_plugins的位置，那么rollout就会卡住，无法启动

# 代码报错
(EngineCore_0 pid=461106) INFO 09-30 17:34:46 [block_pool.py:280] Successfully reset prefix cache
INFO:     127.0.0.1:38354 - "POST /reset_prefix_cache/ HTTP/1.1" 200 OK
  0%|                                                    | 0/32 [00:00<?, ?it/s][WARNING:swift] max_model_len(2048) - num_tokens(1605) < max_tokens(1024). Setting max_tokens: 443
[WARNING:swift] max_model_len(2048) - num_tokens(5221) < max_tokens(1024). Setting max_tokens: -3173
[WARNING:swift] max_model_len(2048) - num_tokens(1605) < max_tokens(1024). Setting max_tokens: 443
[ERROR:swift] Method execution failed: async_infer
Traceback (most recent call last):
  File "/workspace/verl/ms-swift/swift/llm/infer/rollout.py", line 144, in async_llm_worker
    result = await method(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/plugin/multi_turn.py", line 90, in async_infer
    results = await self.infer_engine._batch_infer_stream(tasks, request_config.stream, use_tqdm, None)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/llm/infer/infer_engine/grpo_vllm_engine.py", line 155, in _batch_infer_stream
    return await self.batch_run(new_tasks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/llm/infer/infer_engine/infer_engine.py", line 114, in batch_run
    return await asyncio.gather(*tasks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/tasks.py", line 277, in __step
    result = coro.send(None)
             ^^^^^^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/llm/infer/infer_engine/grpo_vllm_engine.py", line 145, in _new_run
    res = await task
          ^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/plugin/multi_turn.py", line 84, in _infer_async_single
    return await self.run(infer_request, request_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/plugin/multi_turn.py", line 229, in run
    response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/llm/infer/infer_engine/vllm_engine.py", line 743, in infer_async
    generation_config = self._prepare_generation_config(request_config)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/llm/infer/infer_engine/vllm_engine.py", line 406, in _prepare_generation_config
    res = SamplingParams(**kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/vllm/sampling_params.py", line 342, in __post_init__
    self._verify_args()
  File "/usr/local/lib/python3.11/site-packages/vllm/sampling_params.py", line 397, in _verify_args
    raise ValueError(
ValueError: max_tokens must be at least 1, got -3173.
[WARNING:swift] max_model_len(2048) - num_tokens(1603) < max_tokens(1024). Setting max_tokens: 445
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/uvicorn/protocols/http/httptools_impl.py", line 409, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/fastapi/applications.py", line 1054, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.11/site-packages/starlette/applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.11/site-packages/starlette/middleware/errors.py", line 186, in __call__
    raise exc
  File "/usr/local/lib/python3.11/site-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "/usr/local/lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/usr/local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 290, in handle
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 78, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "/usr/local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 75, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/fastapi/routing.py", line 302, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/fastapi/routing.py", line 213, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/ms-swift/swift/llm/infer/rollout.py", line 386, in infer
    all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'NoneType' object is not iterable