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

