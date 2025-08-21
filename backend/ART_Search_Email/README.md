# 单Agent多工具训练

## 准备数据
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2
python train_email_search_agent.py build-db

# 训练
python train_email_search_agent.py train

## kill掉进程
ps aux | grep train_email_search_agent.py | grep -v grep | awk '{print $2}' | xargs kill -9
# 杀掉历史模型占用
pkill -9 model-service

## Train log
[RULER-DEBUG] Sending 2 messages to model `o4-mini`.
13:27:23 - LiteLLM:INFO: utils.py:2958 -
LiteLLM completion() model= o4-mini; provider = openai
[2025-08-21 13:27:23] INFO utils.py:2958:
LiteLLM completion() model= o4-mini; provider = openai
[2025-08-21 13:27:23] INFO _client.py:1740: HTTP Request: POST http://localhost:6688/chat/completions "HTTP/1.1 200 OK"
13:27:34 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: openai/o4-mini-2025-04-16
[2025-08-21 13:27:34] INFO cost_calculator.py:655: selected model name for cost calculation: openai/o4-mini-2025-04-16
[RULER-DEBUG] Got response with 1 choices.
[RULER-DEBUG] Raw model content: {"scores":[{"trajectory_id":"1","explanation":"Directly called read_email without any prior search and
never returned an answer—minimal progress.","score":0.1},{"trajectory_id":"2","explanation":"Correctly initiated a search with relevant
keywords but stopped before reading an email or returning a result.","score":0.4},{"trajectory_id":"3","explanation":"Attempted to read an
email then guessed a value despite no content—some effort but incorrect and no valid
data.","score":0.2},{"trajectory_id":"4","explanation":"Issued a malformed search with wrong keywords and never provided an answer—no
meaningful progress.","score":0.0}]}
[RULER-DEBUG] Pretty-printed JSON:
{
    'scores': [
        {
            'trajectory_id': '1',
            'explanation': 'Directly called read_email without any prior search and never returned an answer—minimal progress.',
            'score': 0.1
        },
        {
            'trajectory_id': '2',
            'explanation': 'Correctly initiated a search with relevant keywords but stopped before reading an email or returning a
result.',
            'score': 0.4
        },
        {
            'trajectory_id': '3',
            'explanation': 'Attempted to read an email then guessed a value despite no content—some effort but incorrect and no valid
data.',
            'score': 0.2
        },
        {
            'trajectory_id': '4',
            'explanation': 'Issued a malformed search with wrong keywords and never provided an answer—no meaningful progress.',
            'score': 0.0
        }
    ]
}
[RULER-DEBUG] Parsed 4 scores from model response.
[RULER-DEBUG] Trajectory updated with score=0.1, explanation=Directly called read_email without any prior search and never returned an
answer—minimal progress.
[RULER-DEBUG] Trajectory updated with score=0.4, explanation=Correctly initiated a search with relevant keywords but stopped before reading
an email or returning a result.
[RULER-DEBUG] Trajectory updated with score=0.2, explanation=Attempted to read an email then guessed a value despite no content—some effort
but incorrect and no valid data.
[RULER-DEBUG] Trajectory updated with score=0.0, explanation=Issued a malformed search with wrong keywords and never provided an answer—no
meaningful progress.
[RULER-DEBUG] Returning new TrajectoryGroup with updated scores.
[DEBUG] TrajectoryGroup.__init__ started
[DEBUG] TrajectoryGroup.__init__ finished
[DEBUG] TrajectoryGroup.__init__ started
[DEBUG] TrajectoryGroup.__init__ finished
[RULER-DEBUG] Scoring trajectory group with 4 trajectories.
[RULER-DEBUG] Created deep copies of trajectories.
[RULER-DEBUG] Extracted 4 message lists for scoring.
[RULER-DEBUG] Received 4 trajectories to score.
[RULER-DEBUG] Common prefix length: 2
[RULER-DEBUG] Common prefix messages: [{"content": "\nYou are an email search agent. Use the provided tools to search and read emails.\nYou
may take up to 10 turns.\nUser's email address is joe.hartsoe@enron.com\nToday's date is 1999-12-11\n", "role": "system"}, {"content":
"When did Steve Whitley say TVA would implement the NERC letter?", "role": "user"}]
[RULER-DEBUG] Serialized trajectory 1: <trajectory id="1">
[{"role": "assistant", "content": null, "tool_calls": [{"id": "chatcmpl-tool-c46503bc01ea41a28c0c4bf63cf8c17e", "type": "function",
"function": {"name": "search_inbox", "arguments": "{\"keywords\": [\"TVA\", \"NERC letter\", \"Steve\"], \"message_id\": \"\"}"}}]}]
</trajectory>
[RULER-DEBUG] Serialized trajectory 2: <trajectory id="2">
[{"role": "assistant", "content": null, "tool_calls": [{"id": "chatcmpl-tool-26c8c68af5cf4c7689236f076784ec60", "type": "function",
"function": {"name": "search_inbox", "arguments": "{\"keywords\": [\"TVA\", \"NERC letter\", \"20th anniversary\", \"TVA strike\"],
\"toolbench_rapid_search\": true, \"max_results\": 5}"}}]}]
</trajectory>
[RULER-DEBUG] Serialized trajectory 3: <trajectory id="3">
[{"role": "assistant", "content": null, "tool_calls": [{"id": "chatcmpl-tool-7ff1cae410644dfa9b376a20b02bd8e1", "type": "function",
"function": {"name": "search_inbox", "arguments": "{\"keywords\": [\"TVA\", \"NERC letter\", \"Steve Whitley\"]}"}}]}, {"content":
"[{'message_id': '<18156447.1075846144018.JavaMail.evans@thyme>', 'snippet': '<b>Steve Whitley</b> told Dick Ingersoll yesterday that
<b>TVA</b> would implement the <b>NERC \\nletter</b> by Monday ... '}]", "role": "tool", "tool_call_id":
"chatcmpl-tool-7ff1cae410644dfa9b376a20b02bd8e1"}, {"role": "assistant", "content": "Steve Whitley told Dick Ingersoll yesterday that TVA
would implement the NERC letter by Monday."}]
</trajectory>
[RULER-DEBUG] Serialized trajectory 4: <trajectory id="4">
[{"role": "assistant", "content": null, "tool_calls": [{"id": "chatcmpl-tool-e014ca4862424feea0e943228db57df4", "type": "function",
"function": {"name": "search_inbox", "arguments": "{\"keywords\": [\"TVA\", \"NERC letter\", \"TVA\", \"implementation\", \"1999-12-11\",
\"TVA\", \"Steve Whitley\", \"20th Century Fund\", \"1998-03-07\"]}"}}]}, {"content": "[]", "role": "tool", "tool_call_id":
"chatcmpl-tool-e014ca4862424feea0e943228db57df4"}, {"role": "assistant", "content": "I was unable to find any specific messages related to
Steve Whitley mentioning TVA implementing the NERC letter on December 11, 1999. If you have a different keyword or topic, let me know, and
I may be able to provide assistance with your query."}]
</trajectory>

[DEBUG] Metrics recorded for future #0
[GatherContext.update_pbar] Called with n=1
[GatherContext.update_pbar] Incrementing pbar by 1
                                                                                                                                           [GatherContext.update_pbar] Metric reward: sum=0.0, divisor=1, avg=0.0                                       | 1/8 [00:06<00:43,  6.24s/it]
[GatherContext.update_pbar] Metric completion_tokens: sum=35.0, divisor=1, avg=35.0
[GatherContext.update_pbar] Setting postfix: {'reward': 0.0, 'completion_tokens': 35.0}
                                                                                                                                           [DEBUG] Progress bar updated (success), total trajectories=1                 | 1/8 [00:06<00:43,  6.24s/it, reward=0, completion_tokens=35]
[DEBUG] Waiting for future #1 ...
[DEBUG] Future #1 completed successfully: Trajectory(reward=0.0, metrics={}, metadata={'scenario_id': 5138, 'step': 10})
[DEBUG] Metrics recorded for future #1
[GatherContext.update_pbar] Called with n=1
[GatherContext.update_pbar] Incrementing pbar by 1
[GatherContext.update_pbar] Metric reward: sum=0.0, divisor=2, avg=0.0
[GatherContext.update_pbar] Metric completion_tokens: sum=77.0, divisor=2, avg=38.5
[GatherContext.update_pbar] Setting postfix: {'reward': 0.0, 'completion_tokens': 38.5}
                                                                                                                                           [DEBUG] Progress bar updated (success), total trajectories=2               | 2/8 [00:06<00:37,  6.24s/it, reward=0, completion_tokens=38.5]
[DEBUG] Waiting for future #2 ...
13:30:17 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: hosted_vllm/email-agent-001
[2025-08-21 13:30:17] INFO cost_calculator.py:655: selected model name for cost calculation: hosted_vllm/email-agent-001
13:30:17 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: email-agent-001
[2025-08-21 13:30:17] INFO cost_calculator.py:655: selected model name for cost calculation: email-agent-001
[2025-08-21 13:30:19] INFO _client.py:1740: HTTP Request: POST http://0.0.0.0:8000/v1/chat/completions "HTTP/1.1 200 OK"
13:30:19 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: hosted_vllm/email-agent-001
[2025-08-21 13:30:19] INFO cost_calculator.py:655: selected model name for cost calculation: hosted_vllm/email-agent-001
13:30:19 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: hosted_vllm/email-agent-001
[2025-08-21 13:30:19] INFO cost_calculator.py:655: selected model name for cost calculation: hosted_vllm/email-agent-001
Error parsing tool calls: rollout.<locals>.search_inbox() got an unexpected keyword argument 'message_ids'
13:30:19 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: hosted_vllm/email-agent-001
[2025-08-21 13:30:19] INFO cost_calculator.py:655: selected model name for cost calculation: hosted_vllm/email-agent-001
13:30:19 - LiteLLM:INFO: cost_calculator.py:655 - selected model name for cost calculation: email-agent-001
[2025-08-21 13:30:19] INFO cost_calculator.py:655: selected model name for cost calculation: email-agent-001
[DEBUG] Future #0 completed successfully: Trajectory(reward=0.0, metrics={}, metadata={'scenario_id': 3653, 'step': 10})
[DEBUG] Metrics recorded for future #0
[GatherContext.update_pbar] Called with n=1
[GatherContext.update_pbar] Incrementing pbar by 1
                                                                                                                                           [GatherContext.update_pbar] Metric reward: sum=0.0, divisor=3, avg=0.0     | 3/8 [00:08<00:11,  2.38s/it, reward=0, completion_tokens=38.5]
[GatherContext.update_pbar] Metric completion_tokens: sum=128.0, divisor=3, avg=42.666666666666664
[GatherContext.update_pbar] Setting postfix: {'reward': 0.0, 'completion_tokens': 42.666666666666664}
                                                                                                                                           [DEBUG] Progress bar updated (success), total trajectories=1               | 3/8 [00:08<00:11,  2.38s/it, reward=0, completion_tokens=42.7]
[DEBUG] Waiting for future #1 ...

divisor 就是算平均值时用的“分母”。
在你的日志里，GatherContext.update_pbar 对每个指标维护三元组
sum / divisor = avg。
每处理完一个 future/trajectory（一次 rollout 结果），就会调用一次 update_pbar（日志里写着 Called with n=1），把本次的度量加到 sum，同时把计数（或权重）加到 divisor。所以：

第一次：reward: sum=0.0, divisor=1, avg=0.0
目前只聚合了 1 条样本，所以分母是 1。

第二次：completion_tokens: sum=77.0, divisor=2, avg=38.5
现在聚合了 2 条样本，平均值 = 77 / 2 = 38.5。由此也能看出第二条样本贡献了 77-35=42 个 tokens。

第三次：completion_tokens: sum=128.0, divisor=3, avg≈42.67
分母继续随样本数增加。



# 测试时日志
python test_email_search_agent.py

Registering model...，进行super中..
调用backend的_prepare_backend_for_training
INFO 08-21 15:52:40 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 08-21 15:52:41 [__init__.py:239] Automatically detected platform cuda.
⚙️  Running in WANDB offline mode
_prepare_backend_for_training启动OpenAI服务: None
[start_openai_server] 开始启动 OpenAI server...
[start_openai_server] 当前 output_dir: ./.art/email-search-agent/models/email-agent-001
[start_openai_server] 传入 config: None
[start_openai_server] get_last_checkpoint_dir 返回: ./.art/email-search-agent/models/email-agent-001/checkpoints/0021
[start_openai_server] 停止可能已有的 OpenAI server...
[start_openai_server] 旧的 OpenAI server 已停止
[start_openai_server] 准备启动新的 openai_server_task，配置如下：
  - model_name: email-agent-001
  - base_model: Qwen/Qwen2.5-0.5B-Instruct
  - log_file: ./.art/email-search-agent/models/email-agent-001/logs/vllm.log
  - lora_path: ./.art/email-search-agent/models/email-agent-001/checkpoints/0021
  - config: {'log_file': './.art/email-search-agent/models/email-agent-001/logs/vllm.log', 'server_args': {'api_key': 'default', 'lora_modules': ['{"name": "email-agent-001", "path": "./.art/email-search-agent/models/email-agent-001/checkpoints/0021"}'], 'return_tokens_as_token_ids': True, 'enable_auto_tool_choice': True, 'tool_call_parser': 'hermes'}, 'engine_args': {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'num_scheduler_steps': 16, 'served_model_name': 'Qwen/Qwen2.5-0.5B-Instruct', 'disable_log_requests': True, 'generation_config': 'vllm'}}
/workspace/verl/ART/src/art/unsloth/state.py:11: UserWarning: WARNING: Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  import unsloth  # type: ignore
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Unsloth: Patching vLLM v1 graph capture
Unsloth: Patching vLLM v0 graph capture
==((====))==  Unsloth 2025.8.6: Fast Qwen2 patching. Transformers: 4.55.2. vLLM: 0.8.5.post1.
   \\   /|    NVIDIA GeForce RTX 4090 D. Num GPUs = 3. Max memory: 23.546 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.6.0+cu124. CUDA: 8.9. CUDA Toolkit: 12.4. Triton: 3.2.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post2. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!