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


## gsm8k.py
```
GSM8K的原始数据:
Example 1:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Example 2:
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Example 3:
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.

处理后的数据
Dataset Info:
Train dataset size: 7473
Test dataset size: 1319

First 3 training examples:

Example 1:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".', 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '72', 'style': 'rule'}, 'extra_info': {'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72', 'index': 0, 'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 'split': 'train'}}

Example 2:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let\'s think step by step and output the final answer after "####".', 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '10', 'style': 'rule'}, 'extra_info': {'answer': 'Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10', 'index': 1, 'question': 'Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?', 'split': 'train'}}

Example 3:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? Let\'s think step by step and output the final answer after "####".', 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '5', 'style': 'rule'}, 'extra_info': {'answer': "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5", 'index': 2, 'question': 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?', 'split': 'train'}}

```

## gsm8k_multiturn_w_interaction.py
多了一个system的内容还有interaction_kwargs
```
Dataset Info:
Train dataset size: 7473
Test dataset size: 1319

First 3 training examples:

Example 1:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'You are a math expert. You are given a question and you need to solve it step by step. You should rethinking carefully if user point out your answer is wrong. Put your final answer in the format of `#### <answer>`.', 'role': 'system'}, {'content': "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after `####`.", 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '72', 'style': 'rule'}, 'extra_info': {'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72', 'index': 0, 'interaction_kwargs': {'ground_truth': '72', 'name': 'gsm8k', 'query': "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after `####`."}, 'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 'split': 'train'}}

Example 2:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'You are a math expert. You are given a question and you need to solve it step by step. You should rethinking carefully if user point out your answer is wrong. Put your final answer in the format of `#### <answer>`.', 'role': 'system'}, {'content': "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let's think step by step and output the final answer after `####`.", 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '10', 'style': 'rule'}, 'extra_info': {'answer': 'Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10', 'index': 1, 'interaction_kwargs': {'ground_truth': '10', 'name': 'gsm8k', 'query': "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let's think step by step and output the final answer after `####`."}, 'question': 'Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?', 'split': 'train'}}

Example 3:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'You are a math expert. You are given a question and you need to solve it step by step. You should rethinking carefully if user point out your answer is wrong. Put your final answer in the format of `#### <answer>`.', 'role': 'system'}, {'content': "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? Let's think step by step and output the final answer after `####`.", 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '5', 'style': 'rule'}, 'extra_info': {'answer': "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5", 'index': 2, 'interaction_kwargs': {'ground_truth': '5', 'name': 'gsm8k', 'query': "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? Let's think step by step and output the final answer after `####`."}, 'question': 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?', 'split': 'train'}}
Creating parquet from Arrow format: 100%|████████| 8/8 [00:00<00:00, 185.08ba/s]
Creating parquet from Arrow format: 100%|████████| 2/2 [00:00<00:00, 243.61ba/s]
```

# gsm8k_multiturn_w_tool.py
对比gsm8k_multiturn_w_interaction多了一个calc_gsm8k_reward工具，系统提示词添加"You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
```
Dataset Info:
Train dataset size: 7473
Test dataset size: 1319

First 3 training examples:

Example 1:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'You are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the `calc_gsm8k_reward` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary. Put your final answer in the format of `#### <answer>`.', 'role': 'system'}, {'content': "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after `####`.", 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '72', 'style': 'rule'}, 'extra_info': {'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72', 'index': 0, 'interaction_kwargs': {'ground_truth': '72', 'query': "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after `####`."}, 'need_tools_kwargs': True, 'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 'split': 'train', 'tools_kwargs': {'calc_gsm8k_reward': {'create_kwargs': {'ground_truth': '72'}}}}}

Example 2:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'You are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the `calc_gsm8k_reward` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary. Put your final answer in the format of `#### <answer>`.', 'role': 'system'}, {'content': "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let's think step by step and output the final answer after `####`.", 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '10', 'style': 'rule'}, 'extra_info': {'answer': 'Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10', 'index': 1, 'interaction_kwargs': {'ground_truth': '10', 'query': "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let's think step by step and output the final answer after `####`."}, 'need_tools_kwargs': True, 'question': 'Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?', 'split': 'train', 'tools_kwargs': {'calc_gsm8k_reward': {'create_kwargs': {'ground_truth': '10'}}}}}

Example 3:
data: {'data_source': 'openai/gsm8k', 'prompt': [{'content': 'You are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the `calc_gsm8k_reward` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary. Put your final answer in the format of `#### <answer>`.', 'role': 'system'}, {'content': "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? Let's think step by step and output the final answer after `####`.", 'role': 'user'}], 'ability': 'math', 'reward_model': {'ground_truth': '5', 'style': 'rule'}, 'extra_info': {'answer': "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5", 'index': 2, 'interaction_kwargs': {'ground_truth': '5', 'query': "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? Let's think step by step and output the final answer after `####`."}, 'need_tools_kwargs': True, 'question': 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?', 'split': 'train', 'tools_kwargs': {'calc_gsm8k_reward': {'create_kwargs': {'ground_truth': '5'}}}}}

```

# gsm8k_tool_agent_loop.py
对比gsm8k_multiturn_w_tool只多了1个"agent_name": "tool_agent",
