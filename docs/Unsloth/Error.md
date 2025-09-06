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

# 尝试llama factory 容器的的trl==0.9.6
安装：docs/llamafactory/README.md
pip install unsloth

#报错
Traceback (most recent call last):
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 464, in <module>
    main()
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 441, in main
    metrics = run_training(trainer)
              ^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 251, in run_training
    stats = trainer.train()
            ^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/trl/trainer/sft_trainer.py", line 451, in train
    output = super().train(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/trainer.py", line 2328, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 323, in _fast_inner_training_loop
  File "<string>", line 40, in _unsloth_training_step
  File "/opt/conda/lib/python3.11/site-packages/unsloth/models/_utils.py", line 1243, in _unsloth_pre_compute_loss
    outputs = self._old_compute_loss(model, inputs, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/trainer.py", line 4099, in compute_loss
    outputs = model(**inputs)
              ^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/peft/peft_model.py", line 818, in forward
    return self.get_base_model()(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/tools/unsloth_compiled_cache/unsloth_compiled_module_qwen3.py", line 611, in forward
    return Qwen3ForCausalLM_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, cache_position, logits_to_keep, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/opt/conda/lib/python3.11/site-packages/torch/_dynamo/external_utils.py", line 198, in nonrecursive_disable_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/utils/generic.py", line 940, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/tools/unsloth_compiled_cache/unsloth_compiled_module_qwen3.py", line 453, in Qwen3ForCausalLM_forward
    outputs: BaseModelOutputWithPast = self.model(
                                       ^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/utils/generic.py", line 1064, in wrapper
    outputs = func(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 410, in forward
    hidden_states = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/modeling_layers.py", line 93, in __call__
    return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_compile.py", line 53, in inner
    return disable_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 929, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 488, in checkpoint
    return CheckpointFunction.apply(function, preserve, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 576, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/unsloth_zoo/gradient_checkpointing.py", line 475, in forward
    outputs = run_function(*args)
              ^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 260, in forward
    hidden_states, _ = self.self_attn(
                       ^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/tools/unsloth_compiled_cache/unsloth_compiled_module_qwen3.py", line 372, in forward
    return Qwen3Attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_dynamo/external_utils.py", line 198, in nonrecursive_disable_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/tools/unsloth_compiled_cache/unsloth_compiled_module_qwen3.py", line 330, in Qwen3Attention_forward
    attn_output = self.o_proj(attn_output)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/verl/tools/unsloth_compiled_cache/Linear_peft_forward.py", line 56, in unsloth_forward
    result = self.base_layer(x, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
  0%|