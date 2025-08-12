import os
import json
import time
import random
import traceback
from typing import Any, Dict, List
from collections import Counter
from dataclasses import dataclass
from contextlib import asynccontextmanager

import openai
import weave
import mcp.types as types
from dotenv import load_dotenv
from openai import AsyncOpenAI
from art.local import LocalBackend
from art.rewards import ruler_score_group
from art.utils import iterate_dataset
import art
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Load environment variables
load_dotenv()

# Configuration constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SMITHERY_MCP_URL = os.getenv("SMITHERY_MCP_URL", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME = "mcprl-3b-exa"
PROJECT_NAME = "mcp-rl"
MAX_TURNS = 10
NUM_TEST_INPUTS = 8
RULER_MODEL = "openrouter/openai/o4-mini"
INPUT_GENERATION_MODEL = "openai/o4-mini"
MAX_SEQ_LENGTH = 16384
GPU_MEMORY_UTILIZATION = 0.7

TRAINING_CONFIG = {
    "num_training_inputs": 16,
    "groups_per_step": 2,
    "num_epochs": 1,
    "rollouts_per_group": 4,
    "learning_rate": 1e-5,
    "max_training_steps": None,
}

# Logging helpers
DEBUG_LOG = True
LOG_JSON_MAX = 2000

def _ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str, **kv):
    if DEBUG_LOG:
        parts = [f"[{_ts()}] {msg}"]
        if kv:
            kv_str = " ".join(f"{k}={repr(v)}" for k, v in kv.items())
            parts.append("| " + kv_str)
        print(" ".join(parts))

def log_json(title: str, payload: Any, max_len: int = LOG_JSON_MAX):
    if DEBUG_LOG:
        try:
            s = json.dumps(payload, indent=2, default=str)
        except Exception:
            s = str(payload)
        if len(s) > max_len:
            s = s[:max_len] + "\n... (truncated)"
        print(f"[{_ts()}] {title}:\n{s}")

# MCP session helpers
@asynccontextmanager
async def mcp_session():
    async with streamablehttp_client(SMITHERY_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

async def list_tools_and_resources():
    async with mcp_session() as session:
        tools = await session.list_tools()
        try:
            resources = await session.list_resources()
        except Exception:
            class _Empty:
                resources = []
            resources = _Empty()
        return tools, resources

async def call_mcp_tool(tool_name: str, arguments: dict):
    async with mcp_session() as session:
        return await session.call_tool(tool_name, arguments)

@dataclass
class McpScenario:
    task_description: str
    max_turns: int = MAX_TURNS

def get_content_text(result) -> str:
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

@weave.op()
async def rollout(model: art.Model, scenario: McpScenario, debug: bool = False) -> art.Trajectory:
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"task": scenario.task_description},
        metrics={"task_completed": False, "success": False, "ran_out_of_turns": False},
        scenario=scenario,
    )

    tools_result, _ = await list_tools_and_resources()
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"MCP tool: {tool.name}",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for tool in tools_result.tools
    ]

    tool_schemas.append({
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Complete the task with a summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of accomplishments"}
                },
                "required": ["summary"],
            },
        },
    })

    traj.tools = tool_schemas
    system_prompt = (
        f"You are an MCP (Model Context Protocol) agent.\n\nUse MCP tools to complete your task.\n\n"
        f"When done, call 'complete_task' with a summary. You have {scenario.max_turns} turns."
    )

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please complete this task: {scenario.task_description}"},
    ]

    num_turns = 0
    task_completed = False

    while num_turns < scenario.max_turns and not task_completed:
        num_turns += 1
        try:
            openai_client = AsyncOpenAI(api_key=model.inference_api_key, base_url=model.inference_base_url)
            response = await openai_client.chat.completions.create(
                model=model.inference_model_name or model.name,
                messages=traj.messages(),
                tools=tool_schemas,
                max_completion_tokens=8000,
            )

            choice = response.choices[0]
            msg = choice.message
            traj.messages_and_choices.append(choice)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                    if tool_call.function.name == "complete_task":
                        traj.metrics["task_completed"] = True
                        task_completed = True
                        traj.messages_and_choices.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": "Task marked complete."
                        })
                    else:
                        result = await call_mcp_tool(tool_call.function.name, tool_args)
                        traj.messages_and_choices.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": get_content_text(result)
                        })
            else:
                log("No tool calls returned", turn=num_turns)

        except Exception as e:
            traceback.print_exc()
            traj.logs.append(f"Error in turn {num_turns}: {e}")
            break

    if not task_completed and num_turns == scenario.max_turns:
        traj.metrics["ran_out_of_turns"] = True
    traj.metrics["num_turns"] = num_turns
    return traj.finish()

# =============== Training loop ===============

async def main_train_and_eval():
    """Main entrypoint to run training and evaluation. Wraps top-level awaits so the file
    can be executed from an asyncio event loop or directly with an async runner."""

    # Validate required configuration
    if not SMITHERY_MCP_URL:
        raise ValueError("SMITHERY_MCP_URL must be set in the environment.")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY must be set in the environment.")

    # Initialize Weave/W&B if requested
    if WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        try:
            weave.init(PROJECT_NAME)
        except Exception as e:
            warn(f"Failed to init weave/wandb: {e}")

    random.seed(42)

    # Declare and configure the TrainableModel
    trainable = art.TrainableModel(name=MODEL_NAME, project=PROJECT_NAME, base_model=BASE_MODEL)
    trainable._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(max_seq_length=MAX_SEQ_LENGTH,),
        engine_args=art.dev.EngineArgs(enforce_eager=True, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,),
    )

    # Local backend for artifacts
    backend = LocalBackend(in_process=True, path="./.art")

    # Register model (idempotent)
    await trainable.register(backend)

    print("Model created and registered:")
    print("  Base model:", BASE_MODEL)
    print("  Model name:", MODEL_NAME)
    print("  Project name:", PROJECT_NAME)

    # --- Scenario generation ---
    expected_total = TRAINING_CONFIG["num_training_inputs"] + NUM_TEST_INPUTS
    info(f"Target total scenarios: {expected_total}")

    # Try generating scenarios (uses the OpenRouter / OpenAI model defined earlier)
    max_attempts = 5
    scenarios = None
    for attempt in range(1, max_attempts + 1):
        step(f"Attempt {attempt}/{max_attempts} to generate scenarios ...")
        try:
            scenarios = await generate_scenarios(num_scenarios=expected_total)
            ok("Scenario generation succeeded.")
            break
        except Exception as e:
            warn(f"Scenario generation failed (attempt {attempt}): {e}")
            if attempt < max_attempts:
                time.sleep(min(1.5 * attempt, 6.0))
            else:
                raise

    # Shuffle and split
    random.shuffle(scenarios)
    train_n = TRAINING_CONFIG["num_training_inputs"]
    raw_train_scenarios = scenarios[:train_n]
    raw_val_scenarios = scenarios[train_n:]

    # Build McpScenario objects
    train_scenarios = [McpScenario(task_description=s["task"], max_turns=MAX_TURNS) for s in raw_train_scenarios]
    val_scenarios = [McpScenario(task_description=s["task"], max_turns=MAX_TURNS) for s in raw_val_scenarios]

    # Create dataset iterator using raw scenarios
    start_step = await trainable.get_step()
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=TRAINING_CONFIG["groups_per_step"],
        num_epochs=TRAINING_CONFIG["num_epochs"],
        initial_step=start_step,
    )

    print("Starting training loop...")
    for batch in train_iterator:
        print(f"Training step {batch.step}: Gathering trajectory groups...")

        # Gather trajectory groups concurrently
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(trainable, scenario, False)
                    for _ in range(TRAINING_CONFIG["rollouts_per_group"])
                )
                for scenario in batch.items
            ),
            pbar_desc=f"train gather step {batch.step}",
        )

        # Score groups with RULER
        scored_groups = []
        for group in groups:
            judged = await ruler_score_group(
                group, judge_model=RULER_MODEL, debug=True, swallow_exceptions=True
            )
            scored_groups.append(judged)

        print("Running model.train on scored groups...")
        await trainable.train(
            scored_groups,
            config=art.TrainConfig(learning_rate=TRAINING_CONFIG["learning_rate"]),
        )

        # Optional checkpointing, metrics, or early stop logic could go here

    print("Training finished. Running evaluation on validation scenarios...")

    # Evaluate on validation scenarios
    for i, scenario in enumerate(val_scenarios):
        print(f"Validation {i+1}/{len(val_scenarios)}: {scenario.task_description}")
        traj = await rollout(trainable, scenario)
        messages = traj.messages()
        model_response = messages[-1]["content"] if messages else "(no response)"
        print("Model output (last message):")
        print(model_response)
        print("-" * 60)

    print("Evaluation complete.")

    # Optional: export or load peft/LoRA checkpoint for inspection
    try:
        lora_model_path = f".art/{trainable.project}/models/{trainable.name}/{await trainable.get_step():04d}"
        print("Attempting to instantiate FastLanguageModel from:", lora_model_path)
        peft_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        print("LoRA model loaded (local).")
    except Exception as e:
        warn(f"Could not load LoRA/PEFT model from disk: {e}")

    print("Done. Artifacts are saved under ./.art/")


# If this file is executed directly, run the async main function.
if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main_train_and_eval())
    except Exception as e:
        err(f"Top-level error: {e}")
        raise

