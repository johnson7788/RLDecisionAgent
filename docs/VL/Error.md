==((====))==  Unsloth 2025.8.6: Fast Qwen2_5_Vl patching. Transformers: 4.55.4. vLLM: 0.10.0.
   \\   /|    NVIDIA GeForce RTX 4090 D. Num GPUs = 3. Max memory: 23.546 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.1+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.3.1
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.31. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Traceback (most recent call last):
  File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 354, in <module>
    main()
  File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 350, in main
    train(args)
  File "/workspace/verl/docs/VL/train_qwen_grpo.py", line 184, in train
    model, tokenizer = FastVisionModel.from_pretrained(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/unsloth/models/loader.py", line 840, in from_pretrained
    model, tokenizer = FastBaseModel.from_pretrained(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/unsloth/models/vision.py", line 444, in from_pretrained
    model = auto_model.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/models/auto/auto_factory.py", line 600, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py", line 317, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py", line 4999, in from_pretrained
    model = cls(config, *model_args, **model_kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Qwen2_5_VLForConditionalGeneration.__init__() got an unexpected keyword argument 'fast_inference'