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

# 报错 https://github.com/inclusionAI/AReaL/issues/240
Traceback (most recent call last):
  File "/workspace/verl/RLDecisionAgent/AReaL/examples/lite/gsm8k_grpo.py", line 226, in <module>
    main(sys.argv[1:])
  File "/workspace/verl/RLDecisionAgent/AReaL/examples/lite/gsm8k_grpo.py", line 81, in main
    rollout.initialize(None, ft_spec)
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/engine/sglang_remote.py", line 80, in initialize
    self._wait_for_server(addr_)
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/engine/sglang_remote.py", line 67, in _wait_for_server
    raise RuntimeError("server launch failed")
RuntimeError: server launch failed
Traceback (most recent call last):
  File "/workspace/verl/RLDecisionAgent/AReaL/examples/lite/gsm8k_grpo.py", line 226, in <module>
    main(sys.argv[1:])
  File "/workspace/verl/RLDecisionAgent/AReaL/examples/lite/gsm8k_grpo.py", line 81, in main
    rollout.initialize(None, ft_spec)
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/engine/sglang_remote.py", line 80, in initialize
    self._wait_for_server(addr_)
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/engine/sglang_remote.py", line 67, in _wait_for_server
    raise RuntimeError("server launch failed")
RuntimeError: server launch failed
Traceback (most recent call last):
  File "/workspace/verl/RLDecisionAgent/AReaL/examples/lite/gsm8k_grpo.py", line 226, in <module>
    main(sys.argv[1:])
  File "/workspace/verl/RLDecisionAgent/AReaL/examples/lite/gsm8k_grpo.py", line 81, in main
    rollout.initialize(None, ft_spec)
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/engine/sglang_remote.py", line 80, in initialize
    self._wait_for_server(addr_)
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/engine/sglang_remote.py", line 67, in _wait_for_server
    raise RuntimeError("server launch failed")
RuntimeError: server launch failed
Traceback (most recent call last):
  File "/workspace/verl/RLDecisionAgent/AReaL/examples/lite/gsm8k_grpo.py", line 226, in <module>
    main(sys.argv[1:])
  File "/workspace/verl/RLDecisionAgent/AReaL/examples/lite/gsm8k_grpo.py", line 81, in main
    rollout.initialize(None, ft_spec)
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/engine/sglang_remote.py", line 80, in initialize
    self._wait_for_server(addr_)
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/engine/sglang_remote.py", line 67, in _wait_for_server
    raise RuntimeError("server launch failed")
RuntimeError: server launch failed
W0813 22:08:39.776000 2585 torch/distributed/elastic/multiprocessing/api.py:898] Sending process 2711 closing signal SIGTERM
W0813 22:08:39.778000 2585 torch/distributed/elastic/multiprocessing/api.py:898] Sending process 2713 closing signal SIGTERM
W0813 22:08:39.779000 2585 torch/distributed/elastic/multiprocessing/api.py:898] Sending process 2714 closing signal SIGTERM
E0813 22:08:39.893000 2585 torch/distributed/elastic/multiprocessing/api.py:870] failed (exitcode: 1) local_rank: 1 (pid: 2712) of binary: /usr/bin/python
Traceback (most recent call last):
  File "/usr/local/bin/torchrun", line 33, in <module>
    sys.exit(load_entry_point('torch==2.7.0a0+ecf3bae40a.nv25.2', 'console_scripts', 'torchrun')())
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/launcher/api.py", line 139, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/launcher/api.py", line 270, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
examples/lite/gsm8k_grpo.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-08-13_22:08:39
  host      : ubuntu22.04
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 2712)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
20250813-22:08:42.218 Local Scheduler INFO: Stopping local process with signal SIGTERM, pid: [2515, 2516, 2517, 2518]
[2025-08-13 22:08:42] Child process unexpectedly failed with exitcode=9. pid=2969
[2025-08-13 22:08:42] Child process unexpectedly failed with exitcode=9. pid=2725
[2025-08-13 22:08:42] Child process unexpectedly failed with exitcode=9. pid=2893
[2025-08-13 22:08:42] Child process unexpectedly failed with exitcode=9. pid=2723
[2025-08-13 22:08:43] Child process unexpectedly failed with exitcode=9. pid=2895
[2025-08-13 22:08:43] Child process unexpectedly failed with exitcode=9. pid=2729
[2025-08-13 22:08:43] Child process unexpectedly failed with exitcode=9. pid=2890
[2025-08-13 22:08:43] Child process unexpectedly failed with exitcode=9. pid=2727
20250813-22:08:43.468 Local Scheduler INFO: Stopping local process with signal SIGTERM, pid: [2520]
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/launcher/local.py", line 304, in <module>
    main_local()
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/launcher/local.py", line 300, in main_local
    raise e
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/launcher/local.py", line 289, in main_local
    launcher.wait(
  File "/workspace/verl/RLDecisionAgent/AReaL/areal/launcher/local.py", line 214, in wait
    raise JobException(
realhf.scheduler.client.JobException: Job gsm8k-grpo_trial0:trainer JobState.COMPLETED at node local
20250813-22:08:44.627 Local Scheduler INFO: Waiting for 0 local running processes, pids: