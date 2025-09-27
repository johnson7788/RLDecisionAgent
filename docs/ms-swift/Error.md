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