# 目录
```
original_model.py  #测试一个模型，加载MCP工具
inference_tool_sft.py  #推理1个lora模型
merge_lora.py  #合并lora模型
prompt.py  #提示词
mcp_config.json  # 使用的MCP工具
mcp_config_load.py #加载mcp工具
```


## Debug方式
1） 在容器里面运行[start_sshd.sh](start_sshd.sh)
2） 然后使用pycharm的ssh连接容器
3） 运行grpo_main.py进行debug