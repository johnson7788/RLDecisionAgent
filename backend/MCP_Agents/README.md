# 目录
```
├── README.md
├── a2a_agent  #Agent，调用MCP生成SFT数据
├── mcpserver  #MCP Server
└── questions.txt  # 生成的问题
└── generate_train_data.py  # 根据生成的问题调用MCP生成SFT数据
└── train.jsonl  # generate_train_data.py生成的训练数据
└── example_one_data.json  # 一条示例数据
```

# 安装依赖
pip install -r a2a_agent/requirements.txt

# 步骤
1）写自己的MCP工具Server，仿照mcpserver/energy_services.py
2）测试MCP工具是正常的, mcpserver/mcp_client.py
3) 生成问题数据，复制你的MCP工具到ChatGPT，让它生成问题数据列表，保存到questions.txt
4) 生成SFT的微调数据：先运行MCP Server, 然后运行a2a_agent/main.py, 然后运行generate_train_data.py生成SFT训练数据
5) 使用生成的训练数据微调模型, python train_tool_sft.py --data_path ./train.jsonl --epochs 3 --lr 2e-4 --batch_size 8 --grad_accum 2 --wandb_project toolsft
6) 测试微调后的模型, python inference_tool_sft.py \
  --model ./lora_model \
  --base_model unsloth/Qwen3-4B-Instruct-2507 \
  --engine unsloth \
  --query "上海今天的天气如何？" \
  --chat_template qwen-3 \
  --load_in_4bit

# 单条SFT的训练数据，带工具
{
    "conversations": [
        {
            "from": "human",
            "value": "你好，我需要一个1到100之间的随机数。"
        },
        {
            "from": "function_call",
            "value": "{\"name\": \"generate_random_number\", \"arguments\": {\"min\": 1, \"max\": 100}}"
        },
        {
            "from": "observation",
            "value": "{\"number\": 57}"
        },
        {
            "from": "gpt",
            "value": "生成的随机数在1到100之间，是57。"
        },
        {
            "from": "human",
            "value": "好的，可以。这次生成一个长度在200到300之间的句子。"
        },
        {
            "from": "function_call",
            "value": "{\"name\": \"generate_random_number\", \"arguments\": {\"min\": 200, \"max\": 300}}"
        },
        {
            "from": "observation",
            "value": "{\"number\": 267}"
        },
        {
            "from": "gpt",
            "value": "生成的随机数在200到300之间，是267。"
        },
        {
            "from": "human",
            "value": "谢谢，这些就是我需要的全部。"
        },
        {
            "from": "gpt",
            "value": "不客气！如果你还需要其他什么，随时问。"
        }
    ],
    "tools": [
        {
            "name": "generate_random_number",
            "description": "在指定范围内生成一个随机数",
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "最小值"
                    },
                    "max": {
                        "type": "integer",
                        "description": "最大值"
                    }
                },
                "required": [
                    "min",
                    "max"
                ]
            }
        }
    ]
}