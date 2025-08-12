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