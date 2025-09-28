# 显卡2用于训练时的推理，自动同步参数，使用SFT过后的模型
export CUDA_VISIBLE_DEVICES=2 \
swift rollout \
    --model ./output/merged_qwen3 \
    --vllm_use_async_engine true \
    --multi_turn_scheduler mcp_call_scheduler \
    --vllm_max_model_len 32768 \
    --vllm_gpu_memory_utilization 0.8 \
    --max_turns 5

# 显卡1用于训练, cd ms-swift目录下，然后运行GRPO， 使用lora训练, train_type 可以用full，表示完全微调
export CUDA_VISIBLE_DEVICES=1
swift rlhf \
    --rlhf_type grpo \
    --model ./output/merged_qwen3 \
    --external_plugins ./plugin.py \
    --reward_funcs external_countdown format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --dataset './train.jsonl' \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --eval_steps 500 \
    --save_steps 100 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --output_dir output/mcp_qwen3 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard \
    --beta 0.001 \
    --num_iterations 1