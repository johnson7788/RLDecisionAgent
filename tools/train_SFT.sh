python train_unsloth_qwen_SFT.py \
    --model_name unsloth/Qwen3-4B-Instruct-2507 \
    --dataset_name mlabonne/FineTome-100k \
    --max_steps 60 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --save_dir ./lora_model