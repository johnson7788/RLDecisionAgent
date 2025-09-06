# bash train_SFT.sh报错，发现是trl版本问题，trl==0.21.0好像没有tokenizer选项
改成processing_class等于tokenizer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
Traceback (most recent call last):
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 461, in <module>
    main()
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 432, in main
    trainer = build_trainer(args, model, tokenizer, dataset)
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 229, in build_trainer
    trainer = SFTTrainer(
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'

# bash train_SFT.sh报错，发现也是trl版本问题，trl==0.21.0
Traceback (most recent call last):
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 461, in <module>
    main()
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 432, in main
    trainer = build_trainer(args, model, tokenizer, dataset)
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 229, in build_trainer
    trainer = SFTTrainer(
  File "/usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py", line 380, in __init__
    raise ValueError(
ValueError: The specified `eos_token` ('<EOS_TOKEN>') is not found in the vocabulary of the given `processing_class` (Qwen2TokenizerFast). Ensure that the `eos_token` exists in the vocabulary before using it as an EOS token.

# FlashInfer报错
安装对应版本或者先临时禁用查看：https://flashinfer.ai/whl的版本
临时禁用，强制 vLLM 用 FlashAttention，绕过 FlashInfer： export VLLM_ATTENTION_BACKEND=FLASH_ATTN 
INFO 09-07 07:06:56 [parallel_state.py:1134] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
WARNING 09-07 07:06:56 [topk_topp_sampler.py:61] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
INFO 09-07 07:06:56 [gpu_model_runner.py:1953] Starting to load model unsloth/Qwen3-4B-Base...
INFO 09-07 07:06:56 [gpu_model_runner.py:1985] Loading model from scratch...
INFO 09-07 07:06:57 [cuda.py:275] Using FlashInfer backend on V1 engine.
[rank0]: Traceback (most recent call last):
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/flashinfer/jit/__init__.py", line 56, in <module>
[rank0]:     from .. import flashinfer_kernels, flashinfer_kernels_sm90  # noqa: F401
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: ImportError: /opt/conda/lib/python3.11/site-packages/flashinfer/flashinfer_kernels.abi3.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb

[rank0]: The above exception was the direct cause of the following exception:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth_zoo/vllm_utils.py", line 1555, in load_vllm
[rank0]:     llm = LLM(**engine_args)
[rank0]:           ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/entrypoints/llm.py", line 285, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/engine/llm_engine.py", line 490, in from_engine_args
[rank0]:     return engine_cls.from_vllm_config(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/llm_engine.py", line 127, in from_vllm_config
[rank0]:     return cls(vllm_config=vllm_config,
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/llm_engine.py", line 104, in __init__
[rank0]:     self.engine_core = EngineCoreClient.make_client(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 82, in make_client
[rank0]:     return InprocClient(vllm_config, executor_class, log_stats)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 245, in __init__
[rank0]:     self.engine_core = EngineCore(*args, **kwargs)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 80, in __init__
[rank0]:     self.model_executor = executor_class(vllm_config)
[rank0]:                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/executor/executor_base.py", line 54, in __init__
[rank0]:     self._init_executor()
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/executor/uniproc_executor.py", line 49, in _init_executor
[rank0]:     self.collective_rpc("load_model")
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/executor/uniproc_executor.py", line 58, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/utils/__init__.py", line 3007, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/worker/gpu_worker.py", line 212, in load_model
[rank0]:     self.model_runner.load_model(eep_scale_up=eep_scale_up)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/worker/gpu_model_runner.py", line 1986, in load_model
[rank0]:     self.model = model_loader.load_model(
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/model_loader/base_loader.py", line 44, in load_model
[rank0]:     model = initialize_model(vllm_config=vllm_config,
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/model_loader/utils.py", line 63, in initialize_model
[rank0]:     return model_class(vllm_config=vllm_config, prefix=prefix)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 287, in __init__
[rank0]:     self.model = Qwen3Model(vllm_config=vllm_config,
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/compilation/decorators.py", line 183, in __init__
[rank0]:     old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 259, in __init__
[rank0]:     super().__init__(vllm_config=vllm_config,
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/compilation/decorators.py", line 183, in __init__
[rank0]:     old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen2.py", line 316, in __init__
[rank0]:     self.start_layer, self.end_layer, self.layers = make_layers(
[rank0]:                                                     ^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/utils.py", line 640, in make_layers
[rank0]:     [PPMissingLayer() for _ in range(start_layer)] + [
[rank0]:                                                      ^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/utils.py", line 641, in <listcomp>
[rank0]:     maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
[rank0]:                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen2.py", line 318, in <lambda>
[rank0]:     lambda prefix: decoder_layer_type(config=config,
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 189, in __init__
[rank0]:     self.self_attn = Qwen3Attention(
[rank0]:                      ^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 123, in __init__
[rank0]:     self.attn = Attention(
[rank0]:                 ^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/attention/layer.py", line 164, in __init__
[rank0]:     self.attn_backend = get_attn_backend(head_size,
[rank0]:                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/attention/selector.py", line 154, in get_attn_backend
[rank0]:     return _cached_get_attn_backend(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/attention/selector.py", line 211, in _cached_get_attn_backend
[rank0]:     return resolve_obj_by_qualname(attention_cls)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/utils/__init__.py", line 2568, in resolve_obj_by_qualname
[rank0]:     module = importlib.import_module(module_name)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank0]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank0]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank0]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank0]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank0]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank0]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/attention/backends/flashinfer.py", line 10, in <module>
[rank0]:     from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/flashinfer/__init__.py", line 18, in <module>
[rank0]:     from .activation import gelu_and_mul as gelu_and_mul
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/flashinfer/activation.py", line 21, in <module>
[rank0]:     from .jit import gen_act_and_mul_module, has_prebuilt_ops, load_cuda_ops    # noqa: F401
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/flashinfer/jit/__init__.py", line 62, in <module>
[rank0]:     raise ImportError("Loading prebuilt ops failed.") from e
[rank0]: ImportError: Loading prebuilt ops failed.

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/docs/Unsloth/unsloth_GRPO.py", line 15, in <module>
[rank0]:     model, tokenizer = FastLanguageModel.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth/models/loader.py", line 404, in from_pretrained
[rank0]:     model, tokenizer = dispatch_model.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth/models/qwen3.py", line 436, in from_pretrained
[rank0]:     return FastLlamaModel.from_pretrained(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth/models/llama.py", line 2088, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth_zoo/vllm_utils.py", line 1578, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError: Loading prebuilt ops failed.
[rank0]:[W907 07:07:00.626827755 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())