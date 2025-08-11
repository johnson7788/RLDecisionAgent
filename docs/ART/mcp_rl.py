# -*- coding: utf-8 -*-
import json
import time
import traceback
from typing import Any
from contextlib import asynccontextmanager
import os
import random
from collections import Counter
from typing import Any, Dict, List
import openai
from dotenv import load_dotenv
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dataclasses import dataclass
import weave
from dotenv import load_dotenv
from openai import AsyncOpenAI

import art
from art.local import LocalBackend
from art.rewards import ruler_score_group
from art.utils import iterate_dataset
import torch
from unsloth import FastLanguageModel


load_dotenv()

# Required - Used for generating training inputs and RULER evaluation
OPENROUTER_API_KEY = "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"  # Put your OpenRouter key here

# 🔌 Point to any Smithery-hosted MCP server (make sure you click "Get URL with keys instead", otherwise this will not work)
SMITHERY_MCP_URL = "https://server.smithery.ai/exa/mcp?api_key=552ddb78-0e95-4998-be87-b936502a5a97&profile=stiff-sole-4FpnEv"

# Optional - Enables metric logging
WANDB_API_KEY = ""

# Choose the base model to train
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Options: "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", etc.

# @title Advanced Settings

# Model configuration
MODEL_NAME = "mcprl-3b-exa"  # Name for your trained model
PROJECT_NAME = "mcp-rl"  # Project name for tracking

# Training configuration
TRAINING_CONFIG = {
    "num_training_inputs": 16,  # Number of training inputs to generate
    "groups_per_step": 2,  # Inputs to process per training step
    "num_epochs": 1,  # Number of times through all data
    "rollouts_per_group": 4,  # Different responses per input (for RULER comparison)
    "learning_rate": 1e-5,  # Learning rate
    "max_training_steps": None,  # Maximum training steps (set to None for no limit)
}

MAX_TURNS = 10  # Maximum number of turns for the model to generate during one rollout

NUM_TEST_INPUTS = 8  # Number of test inputs to generate
RULER_MODEL = "openrouter/openai/o4-mini"  # Model for RULER evaluation
INPUT_GENERATION_MODEL = "openai/o4-mini"

# GPU configuration (for T4 — keep these as-is unless you have a reason to change them)
MAX_SEQ_LENGTH = 16384  # Maximum sequence length
GPU_MEMORY_UTILIZATION = 0.7  # GPU memory usage (0.0-1.0)

DEBUG_LOG = True  # flip to False to silence logs
LOG_JSON_MAX = 2000  # cap large JSON prints


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str, **kv):
    if not DEBUG_LOG:
        return
    parts = [f"[{_ts()}] {msg}"]
    if kv:
        kv_str = " ".join(f"{k}={repr(v)}" for k, v in kv.items())
        parts.append("| " + kv_str)
    print(" ".join(parts))


def log_json(title: str, payload: Any, max_len: int = LOG_JSON_MAX):
    if not DEBUG_LOG:
        return
    try:
        s = json.dumps(payload, indent=2, default=str)
    except Exception:
        s = str(payload)
    if len(s) > max_len:
        s = s[:max_len] + "\n... (truncated)"
    print(f"[{_ts()}] {title}:\n{s}")

if not SMITHERY_MCP_URL:
    raise ValueError("SMITHERY_MCP_URL is empty. Set it in the Configuration cell.")


@asynccontextmanager
async def mcp_session():
    """
    Connects to the remote Smithery MCP server using the full URL that includes
    your API key & profile. No OAuth provider is used.
    """
    async with streamablehttp_client(SMITHERY_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools_and_resources():
    """Return (tools_result, resources_result) from the remote Smithery server."""
    async with mcp_session() as session:
        tools = await session.list_tools()
        try:
            resources = await session.list_resources()
        except Exception:
            # Some servers don't implement resources; keep interface stable
            class _Empty:
                resources = []

            resources = _Empty()
        return tools, resources


async def call_mcp_tool(tool_name: str, arguments: dict):
    """Invoke a tool on the remote Smithery server and return the CallToolResult."""
    async with mcp_session() as session:
        return await session.call_tool(tool_name, arguments)


tools, resources = await list_tools_and_resources()
print("Tools:", [t.name for t in tools.tools])
print(
    "Resources:",
    [getattr(r, "uri", None) for r in getattr(resources, "resources", []) or []],
)

# @title Let's generate our train and validation scenarios!


# ---------- lightweight "nice print" helpers (no extra deps) ----------
class _C:
    RESET = "\x1b[0m"
    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    ITAL = "\x1b[3m"
    GRAY = "\x1b[90m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    MAGENTA = "\x1b[35m"


def _ts():
    return time.strftime("%H:%M:%S")


def info(msg):
    print(f"[{_ts()}] {_C.BLUE}INFO{_C.RESET}  {msg}")


def step(msg):
    print(f"[{_ts()}] {_C.CYAN}STEP{_C.RESET}  {msg}")


def ok(msg):
    print(f"[{_ts()}] {_C.GREEN}OK{_C.RESET}    {msg}")


def warn(msg):
    print(f"[{_ts()}] {_C.YELLOW}WARN{_C.RESET}  {msg}")


def err(msg):
    print(f"[{_ts()}] {_C.RED}ERR{_C.RESET}   {msg}")


def dim(msg):
    print(f"{_C.DIM}{msg}{_C.RESET}")


def preview_scenarios(scenarios, n=5):
    n = min(n, len(scenarios))
    for i in range(n):
        s = scenarios[i]
        dim(
            f"   {i + 1}. {s['task'][:120].strip()}{'…' if len(s['task']) > 120 else ''}  "
            f"{_C.GRAY}(difficulty {s['difficulty']}/5){_C.RESET}"
        )


# ---------- required env/key check ----------
# If OPENROUTER_API_KEY exists as a var, use it; otherwise pull from env
_openrouter_key = os.getenv("OPENROUTER_API_KEY")
try:
    _openrouter_key = _openrouter_key if _openrouter_key else OPENROUTER_API_KEY  # noqa: F821 (defined upstream in your notebook)
except NameError:
    pass

if _openrouter_key:
    os.environ["OPENROUTER_API_KEY"] = _openrouter_key
    ok("OPENROUTER_API_KEY found.")
else:
    err("OPENROUTER_API_KEY is required for data generation and RULER evaluation.")
    raise ValueError(
        "OPENROUTER_API_KEY is required for data generation and RULER evaluation."
    )


# ---------- generator ----------
async def generate_scenarios(
    num_scenarios: int = 24,
) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    step("Fetching MCP tools & resources from remote server …")
    tools_result, resources_result = await list_tools_and_resources()
    ok(f"Fetched tools & resources in {time.perf_counter() - t0:.2f}s.")

    # summarize tools/resources
    try:
        tool_cnt = len(getattr(tools_result, "tools", []) or [])
        res_cnt = len(getattr(resources_result, "resources", []) or [])
    except Exception:
        tool_cnt = res_cnt = 0
    info(f"Available: {tool_cnt} tool(s), {res_cnt} resource(s).")

    tools_info = []
    for tool in tools_result.tools or []:
        tools_info.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        )

    resources_info = []
    for resource in getattr(resources_result, "resources", []) or []:
        resources_info.append(
            {
                "uri": str(resource.uri),
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mimeType,
            }
        )

    step("Preparing prompt & JSON schema …")
    tools_description = json.dumps(tools_info, indent=2)
    resources_description = (
        json.dumps(resources_info, indent=2)
        if resources_info
        else "No resources available"
    )

    prompt = f"""You are an expert at creating realistic scenarios for testing AI agents that interact with MCP (Model Context Protocol) servers.

Given the following available tools and resources from an MCP server, generate {num_scenarios} diverse, realistic scenarios that a user might want to accomplish using these tools.

AVAILABLE TOOLS:
{tools_description}

AVAILABLE RESOURCES:
{resources_description}

Requirements for scenarios:
1. Each scenario should be a task that can be accomplished using the available tools
2. Scenarios should vary in complexity - some simple (1-2 tool calls), some complex (multiple tool calls)
3. Scenarios should cover different use cases and tool combinations (though the task should not specify which tools to use)
4. Each scenario should be realistic - something a real user might actually want to do
5. Assign a difficulty rating from 1 (easy, single tool call) to 5 (hard, complex multi-step analysis)
6. The task should always include generating a summary of the work done and a thorough analysis and report of the results

You must respond with a JSON object containing a "scenarios" array of exactly {num_scenarios} objects. Each object must have:
- "task": string describing the scenario
- "difficulty": integer from 1-5 representing complexity
"""

    response_schema = {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["task", "difficulty"],
                    "additionalProperties": False,
                },
                "minItems": num_scenarios,
                "maxItems": num_scenarios,
            }
        },
        "required": ["scenarios"],
        "additionalProperties": False,
    }

    # OpenRouter client (via OpenAI SDK)
    try:
        model = INPUT_GENERATION_MODEL  # noqa: F821 (defined elsewhere in your notebook)
    except NameError:
        model = "openai/gpt-4.1-mini"  # safe default if not set
        warn(f"INPUT_GENERATION_MODEL not defined; using default: {model}")

    step(f"Calling OpenRouter model: {_C.BOLD}{model}{_C.RESET} …")
    client_openai = openai.OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    t1 = time.perf_counter()
    response = client_openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scenario_list", "schema": response_schema},
        },
    )
    dt = time.perf_counter() - t1
    ok(f"Model responded in {dt:.2f}s.")

    content = response.choices[0].message.content
    info(f"Raw content length: {len(content)} chars.")
    # Parse JSON
    try:
        result = json.loads(content)
    except Exception as e:
        err("Failed to parse JSON from model response.")
        dim(f"   Exception: {e}")
        dim("   First 500 chars of response content:")
        dim(content[:500])
        raise

    # Extract scenarios
    if "scenarios" in result:
        scenarios = result["scenarios"]
    else:
        scenarios = result if isinstance(result, list) else list(result.values())[0]

    # Validate count
    if len(scenarios) != num_scenarios:
        err(f"Expected {num_scenarios} scenarios, got {len(scenarios)}.")
        raise ValueError(f"Expected {num_scenarios} scenarios, got {len(scenarios)}")

    ok(f"Parsed {len(scenarios)} scenario(s) successfully.")
    preview_scenarios(scenarios, n=min(5, num_scenarios))
    return scenarios


# ---------- run generation w/ attempts ----------
try:
    expected_total = TRAINING_CONFIG["num_training_inputs"] + NUM_TEST_INPUTS  # noqa: F821
except NameError:
    err("TRAINING_CONFIG/NUM_TEST_INPUTS not defined in this notebook.")
    raise

info(f"Target total scenarios: {expected_total}")
max_attempts = 10
scenarios = None

for attempt in range(1, max_attempts + 1):
    step(f"Attempt {attempt}/{max_attempts} …")
    t_attempt = time.perf_counter()
    try:
        scenarios = await generate_scenarios(num_scenarios=expected_total)
        ok(f"Attempt {attempt} succeeded in {time.perf_counter() - t_attempt:.2f}s.")
        break
    except Exception as e:
        warn(f"Attempt {attempt} failed: {e}")
        if attempt < max_attempts:
            time.sleep(min(1.5 * attempt, 6.0))
        else:
            err("All attempts exhausted.")
            raise

# ---------- post-process & reporting ----------
print()  # spacing
ok(f"Generated {len(scenarios)} scenarios total.")
info("Difficulty distribution:")
diff_counts = Counter(s["difficulty"] for s in scenarios)
for d in range(1, 6):
    cnt = diff_counts.get(d, 0)
    bar = "█" * min(cnt, 30)
    dim(f"   {d}/5: {cnt:3d}  {bar}")

print()
step("Shuffling scenarios and splitting into train/val …")
random.shuffle(scenarios)

train_n = TRAINING_CONFIG["num_training_inputs"]  # noqa: F821
raw_train_scenarios = scenarios[:train_n]
raw_val_scenarios = scenarios[train_n:]

ok(f"Train: {len(raw_train_scenarios)} | Val: {len(raw_val_scenarios)}")

info("Sample (train) preview:")
preview_scenarios(raw_train_scenarios, n=min(5, len(raw_train_scenarios)))

info("Sample (val) preview:")
preview_scenarios(raw_val_scenarios, n=min(5, len(raw_val_scenarios)))

print()
ok("Done.")

# @title Run this cell to train your model!



# Optional
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    weave.init(PROJECT_NAME)
else:
    print("WANDB_API_KEY is not set. We'll skip logging metrics to Weights & Biases.")

random.seed(42)

# Declare the model
model = art.TrainableModel(
    name=MODEL_NAME,
    project=PROJECT_NAME,
    base_model=BASE_MODEL,
)

# To run on a T4, we need to override some config defaults.
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=MAX_SEQ_LENGTH,
    ),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    ),
)

# Initialize the server
backend = LocalBackend(
    in_process=True,
    path="./.art",
)

# Register the model with the local Backend
await model.register(backend)

print("Model created!")
print("Base model:", BASE_MODEL)
print("Model name:", MODEL_NAME)
print("Project name:", PROJECT_NAME)


def get_content_text(result) -> str:
    # Extract text content from tool call result per MCP content schema
    if isinstance(result, str):
        return result
    if hasattr(result, "content") and result.content:
        out = ""
        for item in result.content:
            if isinstance(item, types.TextContent):
                out += item.text
            else:
                out += str(item)
        return out
    if hasattr(result, "structured") and result.structured is not None:
        try:
            return json.dumps(result.structured)
        except Exception:
            return str(result.structured)
    return str(result)


@dataclass
class McpScenario:
    """A scenario for MCP agent evaluation against a remote Smithery server."""

    task_description: str
    max_turns: int = MAX_TURNS


@weave.op()
async def rollout(
    model: art.Model,
    scenario: McpScenario,
    debug: bool = False,
) -> art.Trajectory:
    """Run an MCP agent rollout against the remote Smithery MCP server."""
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"task": scenario.task_description},
        metrics={
            "task_completed": False,
            "success": False,
            "ran_out_of_turns": False,
        },
        scenario=scenario,
    )

    # Discover available tools from the remote server
    tools_result, _resources_result = await list_tools_and_resources()
    tool_names = [t.name for t in tools_result.tools]
    log("rollout: discovered tools", count=len(tool_names), names=tool_names)

    # Convert to OpenAI tool format
    tool_schemas = []
    for tool in tools_result.tools:
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"MCP tool: {tool.name}",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
        tool_schemas.append(tool_schema)

    # Add completion tool schema
    tool_schemas.append(
        {
            "type": "function",
            "function": {
                "name": "complete_task",
                "description": "Complete the task with a summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of accomplishments",
                        }
                    },
                    "required": ["summary"],
                },
            },
        }
    )

    traj.tools = tool_schemas

    # Initialize conversation
    system_prompt = (
        f"You are an MCP (Model Context Protocol) agent.\n\n"
        f"Use MCP tools through the server to complete your task.\n\n"
        f"When you believe you have completed the task, call the 'complete_task' function with a summary of what you accomplished. "
        f"You have a total of {scenario.max_turns} turns."
        # NOTE: removing 'Only use tool calls, do not write any content.' — some models
        # will freeze if they think plain text is disallowed. Let them output thoughts but
        # we only process tool calls below.
    )

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Please complete this task: {scenario.task_description}",
        },
    ]

    num_turns = 0
    task_completed = False

    # Main interaction loop
    while num_turns < scenario.max_turns and not task_completed:
        num_turns += 1

        try:
            # === Log request ===
            last_user = next(
                (m for m in reversed(traj.messages()) if m["role"] == "user"), None
            )
            log(
                "LLM request",
                step=num_turns,
                model=(model.inference_model_name or model.name),
                tools=len(tool_schemas),
                last_user=(last_user["content"][:160] + "..." if last_user else None),
            )

            # Get LLM response
            async with traj.track_duration("llm_completion"):
                openai_client = AsyncOpenAI(
                    api_key=model.inference_api_key,
                    base_url=model.inference_base_url,
                )

                # We also log the request body (without huge params)
                req_preview = {
                    "model": model.inference_model_name
                    if model.inference_model_name
                    else model.name,
                    "messages_len": len(traj.messages()),
                    "tools_len": len(tool_schemas),
                }
                log_json("LLM request (preview)", req_preview)

                response = await openai_client.chat.completions.create(
                    model=model.inference_model_name
                    if model.inference_model_name
                    else model.name,
                    messages=traj.messages(),
                    tools=tool_schemas,
                    max_completion_tokens=8000,
                )

            # === Log response ===
            choice = response.choices[0]

            finish_reason = getattr(choice, "finish_reason", None)
            msg = choice.message
            has_tools = bool(getattr(msg, "tool_calls", None))
            content_preview = (
                (msg.content[:200] + "...")
                if isinstance(msg.content, str) and msg.content
                else str(msg.content)[:200]
            )
            log(
                "LLM response parsed",
                finish_reason=finish_reason,
                has_tool_calls=has_tools,
                content_preview=content_preview,
            )

            traj.messages_and_choices.append(choice)

            # Handle tool calls
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    try:
                        log(
                            "Tool call received",
                            name=tool_call.function.name,
                            raw_args=tool_call.function.arguments,
                        )
                        tool_args = json.loads(tool_call.function.arguments or "{}")

                        if tool_call.function.name == "complete_task":
                            traj.metrics["task_completed"] = True
                            task_completed = True
                            traj.logs.append(
                                f"Task completion attempted with summary: {tool_args.get('summary', '')}"
                            )
                            # We still append a tool message for completeness
                            traj.messages_and_choices.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": "Task marked complete.",
                                }
                            )
                        else:
                            # 🔧 Call MCP tool through remote Smithery session
                            result = await call_mcp_tool(
                                tool_call.function.name, tool_args
                            )

                            content_text = get_content_text(result)
                            log(
                                "Tool result",
                                name=tool_call.function.name,
                                len=len(content_text),
                            )

                            if len(content_text) > 20000:
                                # print(
                                #     f"Tool call result for {tool_call.function.name} is too long: {len(content_text)}"
                                # )
                                # print(f"Args: {tool_args}")
                                # print(content_text[:1000])
                                # print(content_text[-1000:])
                                raise Exception(
                                    f"Tool call result for {tool_call.function.name} is too long: {len(content_text)}"
                                )

                            # Add tool response
                            traj.messages_and_choices.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": content_text,
                                }
                            )

                    except Exception as e:
                        traceback.print_exc()
                        traj.logs.append(f"Tool call error: {e}")

                        # Add error response
                        traj.messages_and_choices.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {str(e)}",
                            }
                        )
            else:
                # No tool calls — log and continue (RULER will likely give 0)
                log(
                    "LLM returned no tool_calls; skipping tool execution",
                    turn=num_turns,
                )
                # You can consider breaking here or letting it try another turn
                # break

        except Exception as e:
            traceback.print_exc()
            traj.logs.append(f"Error in turn {num_turns}: {e}")
            break

    if not task_completed and num_turns == scenario.max_turns:
        traj.metrics["ran_out_of_turns"] = True

    traj.metrics["num_turns"] = num_turns

    return traj.finish()


# =============== Training code ===============

print(
    f"Using config: max_turns={MAX_TURNS}, rollouts_per_group={TRAINING_CONFIG['rollouts_per_group']}, "
    f"groups_per_step={TRAINING_CONFIG['groups_per_step']}, num_epochs={TRAINING_CONFIG['num_epochs']}, "
    f"learning_rate={TRAINING_CONFIG['learning_rate']}"
)

await model.register(backend)

train_scenarios = [
    McpScenario(
        task_description=scenario["task"],
        max_turns=MAX_TURNS,
    )
    for scenario in raw_train_scenarios
]

# Create dataset iterator using raw scenarios
train_iterator = iterate_dataset(
    train_scenarios,
    groups_per_step=TRAINING_CONFIG["groups_per_step"],
    num_epochs=TRAINING_CONFIG["num_epochs"],
    initial_step=await model.get_step(),  # Resume from checkpoint
)

# Main training loop using iterate_dataset
for batch in train_iterator:
    print("Gathering trajectory groups with RULER scoring...")

    # Use gather_trajectory_groups with ruler_score_group
    groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                rollout(model, scenario, False)
                for _ in range(TRAINING_CONFIG["rollouts_per_group"])
            )
            for scenario in batch.items
        ),
        pbar_desc=f"train gather step {batch.step}",
    )

    scored_groups = []
    for group in groups:
        # Use RULER to assign relative scores to each trajectory
        judged_group = await ruler_score_group(
            group, judge_model=RULER_MODEL, debug=True, swallow_exceptions=True
        )
        scored_groups.append(judged_group)

    print("starting train")
    await model.train(
        scored_groups,
        config=art.TrainConfig(learning_rate=TRAINING_CONFIG["learning_rate"]),
    )

# @title Test Your Model!

# Generate test inputs
print("Generating test inputs...")
val_scenarios = [
    McpScenario(
        task_description=scenario["task"],
        max_turns=MAX_TURNS,
    )
    for scenario in raw_val_scenarios
]

print(f"\n🧪 Testing the trained model on {len(val_scenarios)} new inputs:\n")
print("=" * 80)

for i, scenario in enumerate(val_scenarios):
    print(f"\nTest {i + 1}:")
    print(f"Input: {scenario.task_description}")

    # Run the model
    result_trajectory = await rollout(model, scenario)

    # Extract the model's response
    messages = result_trajectory.messages()
    model_response = messages[-1]["content"] if messages else "No response"

    print(f"Model output: {model_response}")
    print("-" * 80)

print("\n🎉 Testing completed!")
print(
    f"\nYour model '{MODEL_NAME}' has been trained to use the Smithery MCP server at:"
)
print(SMITHERY_MCP_URL)
print("\nTo use this model in production:")
print("1. The model checkpoint is saved in ./.art/")
print("2. You can load it using the vLLM library")
print(
    "3. Or continue training with more examples by adjusting the configuration at the top"
)

# @title Upload to Hugging Face 🤗


lora_model_path = (
    f".art/{model.project}/models/{model.name}/{await model.get_step():04d}"
)

peft_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_model_path,
    max_seq_length=16384,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

if False:  # Change to True to upload finetune
    peft_model.push_to_hub_merged(f"HF_ACCOUNT/{model.name}", tokenizer, token="hf_...")
