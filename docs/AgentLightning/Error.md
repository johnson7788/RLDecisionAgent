# 训练报错：
https://github.com/microsoft/agent-lightning/issues/48
h non-zero code: 1.
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/workspace/verl/agent-lightning/agentlightning/trainer.py", line 166, in _worker_main_loop
    if agent.trained_agents:
AttributeError: 'RAGAgent' object has no attribute 'trained_agents'