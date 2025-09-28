# 需要先启动MCP server
```bash
cd step1
python energy_services.py
```

# 启动微调后的模型
```bash
cd step5
swift deploy \
    --adapters ../output/v1-20250927-221821/checkpoint-2 \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048 \
    --served_model_name 'mylora-model'
```