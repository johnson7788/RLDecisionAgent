import json
import re
from typing import Any

import datasets
from omegaconf import OmegaConf

code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)


def extract_code_message(content: str) -> tuple[dict[str, Any], str]:
    start, stop = "<code>", "</code>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    code = content[i + len(start):j]
    matches = code_pattern.findall(code)
    if matches:
        code = matches[0].strip()

    message = {
        "role": "assistant",
        "content": content[:i].strip(),
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "arguments": {"code": code},
                },
            },
        ],
    }

    print("\n[extract_code_message] Extracted assistant code message:")
    print(json.dumps(message, indent=2))

    return message, content[j + len(stop):]


def extract_answer_message(content: str) -> tuple[dict[str, Any], str]:
    start, stop = "<answer>", "</answer>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    answer = content[:i] + content[i + len(start):j]
    message = {
        "role": "assistant",
        "content": answer.strip(),
    }

    print("\n[extract_answer_message] Extracted assistant final answer:")
    print(json.dumps(message, indent=2))

    return message, content[j + len(stop):]


def extract_interpreter_message(content: str) -> tuple[dict[str, Any], str]:
    start, stop = "<interpreter>", "</interpreter>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    interpreter = content[i + len(start):j]
    message = {
        "role": "tool",
        "content": interpreter.strip(),
    }

    print("\n[extract_interpreter_message] Extracted tool (interpreter) output:")
    print(json.dumps(message, indent=2))

    return message, content[j + len(stop):]


def process(row: dict, *, tools: str):
    messages = []

    # extract problem
    content = row["messages"][0]["content"]
    start = "*user question:*"
    i = content.find(start)
    assert i != -1
    prompt = content[i + len(start):].replace("<answer>", "").replace("</answer>", "").strip()

    user_message = {
        "role": "user",
        "content": prompt,
    }
    messages.append(user_message)

    print("\n[process] New sample --------------------------")
    print("[process] User prompt:")
    print(prompt)

    # extract multi-turns
    content = row["messages"][1]["content"]
    role = "assistant"
    while len(content) > 0:
        if role == "assistant":
            message, content = extract_code_message(content)
            if message is None:
                message, content = extract_answer_message(content)
            assert message is not None
            messages.append(message)
            role = "tool"
        else:
            message, content = extract_interpreter_message(content)
            assert message is not None
            messages.append(message)
            role = "assistant"

    print("[process] Final message sequence:")
    for msg in messages:
        print(json.dumps(msg, indent=2))

    return {"messages": messages, "tools": tools}


if __name__ == "__main__":
    tools_config_file = "recipe/retool/sandbox_fusion_tool_config.yaml"
    tools_config = OmegaConf.load(tools_config_file)
    tool_schema = OmegaConf.to_container(tools_config["tools"][0]["tool_schema"])
    tools = json.dumps([tool_schema])

    print("\n[main] Loaded tool schema:")
    print(json.dumps(json.loads(tools), indent=2))

    print("\n[main] Loading dataset...")
    data = datasets.load_dataset("JoeYing/ReTool-SFT")["train"]

    print(f"[main] Dataset loaded. Number of samples: {len(data)}")

    # 只处理前几个样本用于观察结构
    data = data.select(range(3))

    print("\n[main] Processing dataset...")
    data = data.map(process, fn_kwargs={"tools": tools})

    print("\n[main] Writing to parquet file...")
    data.to_parquet("wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet")
    print("[main] Done.")
