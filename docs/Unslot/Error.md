# bash train_SFT.sh报错，发现是trl版本问题，trl==0.21.0好像没有tokenizer选项
Traceback (most recent call last):
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 461, in <module>
    main()
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 432, in main
    trainer = build_trainer(args, model, tokenizer, dataset)
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 229, in build_trainer
    trainer = SFTTrainer(
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'