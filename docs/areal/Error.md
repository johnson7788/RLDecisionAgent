# 报错
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/usr/local/lib/python3.10/dist-packages/sglang/launch_server.py", line 6, in <module>
    from sglang.srt.entrypoints.http_server import launch_server
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/entrypoints/http_server.py", line 51, in <module>
    from sglang.srt.entrypoints.engine import _launch_subprocesses
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/entrypoints/engine.py", line 41, in <module>
    from sglang.srt.managers.data_parallel_controller import (
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/data_parallel_controller.py", line 32, in <module>
    from sglang.srt.managers.io_struct import (
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/io_struct.py", line 26, in <module>
    from sglang.srt.managers.schedule_batch import BaseFinishReason
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/schedule_batch.py", line 55, in <module>
    from sglang.srt.mem_cache.allocator import (
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/allocator.py", line 30, in <module>
    from sglang.srt.mem_cache.memory_pool import SWAKVPool
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/memory_pool.py", line 16, in <module>
    from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/torch_memory_saver_adapter.py", line 10, in <module>
    _memory_saver = torch_memory_saver.torch_memory_saver
AttributeError: module 'torch_memory_saver' has no attribute 'torch_memory_saver'. Did you mean: 'TorchMemorySaver'?