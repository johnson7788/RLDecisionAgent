# 这几个数据处理脚本的区别
[gsm8k.py](..%2Fverl%2Fexamples%2Fdata_preprocess%2Fgsm8k.py)
[gsm8k_multiturn_w_interaction.py](..%2Fverl%2Fexamples%2Fdata_preprocess%2Fgsm8k_multiturn_w_interaction.py)
[gsm8k_multiturn_w_tool.py](..%2Fverl%2Fexamples%2Fdata_preprocess%2Fgsm8k_multiturn_w_tool.py)
[gsm8k_tool_agent_loop.py](..%2Fverl%2Fexamples%2Fdata_preprocess%2Fgsm8k_tool_agent_loop.py)


| 脚本名                                | 是否加入 System Prompt | 是否使用 Tool | 是否支持 Agent Loop | Prompt 类型             | 特殊字段                                                        |
| ---------------------------------- | ------------------ | --------- | --------------- | --------------------- | ----------------------------------------------------------- |
| `gsm8k.py`                         | ❌ 无                | ❌ 无       | ❌ 无             | 简单单轮 user prompt      | 基础版                                                         |
| `gsm8k_multiturn_w_interaction.py` | ✅ 有                | ❌ 无       | ❌ 无             | 多轮 system+user prompt | `interaction_kwargs`                                        |
| `gsm8k_multiturn_w_tool.py`        | ✅ 有                | ✅ 有       | ❌ 无             | 多轮 system+user prompt | `need_tools_kwargs` + `tools_kwargs` + `interaction_kwargs` |
| `gsm8k_tool_agent_loop.py`         | ✅ 有                | ✅ 有       | ✅ 有             | 多轮 system+user prompt | 多了 `agent_name="tool_agent"`                                |
