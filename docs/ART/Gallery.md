# ModelConfig
ModelConfig 并不是一个具体的类定义，而是一个类型变量（TypeVar）。
ModelConfig = TypeVar("ModelConfig", bound=BaseModel | None)
这里的 bound=BaseModel | None 意味着 ModelConfig 可以是任何继承自 pydantic.BaseModel 的类，或者 None。
1. 灵活性: Model 或 TrainableModel
   的实例可以携带一个具体的配置对象，这个对象的类型不是写死的。你可以定义自己的配置类（只要它继承自
   pydantic.BaseModel），然后将它用于 Model 的 config 字段。
2. 类型安全: 使用泛型和类型变量，静态类型检查工具（如 MyPy）可以知道当你创建一个 Model 实例时，它的 config 属性应该是什么类型。
3. 无具体实现: art 库本身不提供一个名为 ModelConfig 的具体配置类。它只是定义了一个“插槽”或“模板”，让使用者来填充具体的配置。

  总结一下：

* 它是什么？ ModelConfig 是一个类型占位符（泛型类型变量），代表任何用于模型配置、且继承自 pydantic.BaseModel 的类。
* 里面有什么？ 它的内容取决于你或库的其他部分如何定义并传入一个具体的配置类。它本身没有字段。

# Unsloth
Unsloth 是一种优化后的强化学习训练基础设施，实现了 GRPO 等算法。
ART 是基于 Unsloth 的更高层构建，提供代理训练整个 pipeline（轨迹采集、奖励评估、训练循环等）的一体化工具库。

# await model.register(backend)，是运行到这里
ART/src/art/model.py
async def register(
        self,
        model: "Model",
    ) -> None:
        """

# weave 
其实是一个 LLM 训练与推理的可观测性 / 追踪 (observability & tracing) 库，主要用于记录模型的运行信息、日志、指标和调用链。
它是 Weights & Biases (wandb) 旗下的一个项目，定位类似于：
给 LLM 应用 加上自动化的 logging / tracing / metrics 收集；
在 推理 / 训练 / agent rollout 时，把调用链、prompt、response、latency、错误信息等追踪下来；
在 web dashboard 上可视化调用过程，方便调试、监控和复现。


# 测试MCP工具
运行server端
cd backend/ART_mcp-rl/servers/python/mcp_caculator
python server.py --transport sse

配置config.json
cat config.json
{
  "mcpServers": {
    "everything": {
      "type": "sse",
      "url": "http://localhost:8001/sse"
    },
    "my-server": {
      "command": "node",
      "args": ["build/index.js", "arg1", "arg2"],
      "env": {
        "key": "value",
        "key2": "value2"
      }
    }
  }
}
启动测试工具
npx @modelcontextprotocol/inspector --config ./config.json --server everything

# LocalBackend中的in_process参数
in_process 是一个开关参数，用来决定 模型服务（model-service）是直接在当前 Python 进程里运行，还是要 fork / spawn 成一个独立的子进程运行。


# Unsloth加载同等模型时，会自动加载节省显存的更小版本
Unsloth 在“偷偷帮你省显存/提速”：

你在 vLLM 的 engine_args 里把 model 设成了 Qwen/Qwen2.5-0.5B-Instruct，所以「对外宣称」的 base_model 就是它。

但你项目里 引入了 Unsloth（日志里有 Unsloth: Will patch...），Unsloth 会对常见模型做 自动优化与替换：当检测到可用的同等模型的 4bit 预量化权重 时，会把实际加载的权重换成自己在 HF 上的镜像，例如
unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit，并同时把 vLLM 的 quantization=bitsandbytes、load_format=bitsandbytes 等参数一并设置好。
这就是为啥后面 vLLM 的初始化与下载日志显示的是 unsloth/...-bnb-4bit。

简单说：名义上仍是 Qwen 官方模型；实际加载的是 Unsloth 的 4bit 等价权重，这样更省显存、更快，但行为（除了量化误差）与原模型对齐。


# qwen的chat_template.jinja，  chat的jinja模版
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}


# 多卡训练
LocalBackend 类会在你智能体运行的同一台机器上启动一个 vLLM 服务器和一个 Unsloth 或 torchtune 实例。
https://github.com/OpenPipe/ART/pull/163/commits
多 GPU 支持是近期加到 torchtune service；同时有维护者说明“torchtune 做 full finetune，通常用于单节点多 GPU；Unsloth 适合单 GPU 的 LoRA”。
训练后端升级成可分布式的 torchtune service，并在推理侧配合 vLLM，从而让你在一台多 GPU 机器，或通过 SkyPilot 起的多 GPU 节点上做 单机多卡 训练


# 测试兼容的openai
python -m ART.src.art.openai_patch


# 可以考虑在环境变量里面添加IMPORT_UNSLOTH和IMPORT_PEFT，因为ART/src/art/local/backend.py里设置了，让它们提前加载，提升性能
IMPORT_UNSLOTH=1
IMPORT_PEFT=1


# metrics值
exception_rate： 轨迹异常的数量
reward_std_dev：reward 的平均标准差

# Scenario
简短说：**Scenario 就是一次“任务+环境配置”的打包**。它把“要做什么”（task）和“在哪儿/怎么做”（环境或服务器参数、初始状态、步数上限等）装进一个可序列化的小对象里，交给 `rollout(...)` 去运行并采样轨迹（trajectory）。不同子项目里有各自的 Scenario 结构，但本质都是“让 `rollout` 知道如何开局和何时结束”的数据容器。

## 代码里已经有哪些 Scenario？

* **MCP 代理训练（mcp-rl）**
  `McpScenario`：一个 dataclass，字段很精简

  * `task_description: str` —— 要求代理完成的自然语言任务
  * `server_params: StdioServerParameters` —— 要连接的 MCP 服务器配置
  * `max_turns: int = 10` —— 回合上限
    用法：`rollout(model, scenario: McpScenario)` 在给定 MCP 服务器里执行这项任务（见 `mcp_rl/rollout.py` 第 33–45 行与第 215–221 行示例）。

* **邮件问答实验（art-e.py）**
  两层结构：

  * `Scenario(BaseModel)`：数据集里的“问答场景”，含 `id / question / answer / message_ids / how_realistic / inbox_address / query_date / split` 等，用来**定义问题与参考答案**、以及判分需要的元信息（见 `art-e.py` 第 118–127 行；加载函数在第 466–483 行把每条数据转成 `Scenario`）。
  * `EmailScenario(BaseModel)`：训练/推理时的“执行包装”，含 `step` 与 `scenario: Scenario`，交给 `rollout(model, email_scenario)` 使用（见 `art-e.py` 第 641–648、942–946 行）。

* **井字棋自博弈（tic\_tac\_toe*）*\*
  `TicTacToeScenario(BaseModel)`：描述博弈初始条件和训练/验证分割

  * 轻量版仅有 `step`（`tic_tac_toe/rollout.py` 第 27–33 行）。
  * 自博弈版增加 `split / x_teacher / o_teacher / initial_move` 等（`tic_tac_toe_self_play/rollout.py` 第 103–113 行）。
    用法：`rollout(..., scenario=TicTacToeScenario(...))`（多处示例，如 `train.py`/`tic-tac-toe.py`）。

## 把它抽象出来：Scenario 的通用组成

1. **任务**：自然语言目标或游戏/问题定义（`task_description` / `question`）。
2. **环境**：怎么接入外部系统或如何初始化状态（如 `server_params`、棋局初始落子）。
3. **约束**：回合/步数上限、难度、split 等（`max_turns`、`split`、`difficulty`）。
4. **评测信息（可选）**：参考答案、判分所需元数据（`answer`、`message_ids` 等）。

## 我该如何定义自己的 Scenario？

关键是**先看你的 `rollout` 需要什么**。`rollout` 的函数签名决定了 Scenario 的字段。举三个最小模板：

* **针对 MCP 工具调用类任务**（复用现有 `McpScenario` 足够）：

  ```python
  from mcp_rl.mcp_rl.rollout import McpScenario, rollout

  scenario = McpScenario(
      task_description="用 search_symbol 搜 biotech 相关公司并整理结果",
      server_params=server_params,  # 你已有的 MCP StdioServerParameters
      max_turns=8,
  )
  traj = await rollout(model, scenario)
  ```

* **针对“有参考答案”的信息检索/问答**（仿照 `art-e.py`）：

  ```python
  from pydantic import BaseModel
  from typing import List, Literal

  class QARefScenario(BaseModel):
      id: int
      question: str
      answer: str                   # 评测用的参考答案
      evidence_ids: List[str] = []  # 可选：引用到的数据/文档键
      split: Literal["train", "test"] = "train"

  class QARunScenario(BaseModel):
      step: int
      scenario: QARefScenario

  # rollout(model, qa_run_scenario) 内部读取 question、产生答案，再对比 reference
  ```

* **针对“有明确初始状态”的交互/博弈**（仿照井字棋）：

  ```python
  from pydantic import BaseModel
  from typing import Optional

  class GameScenario(BaseModel):
      step: int
      split: str = "train"
      seed: Optional[int] = None
      initial_state: Optional[dict] = None
  ```

## 设计小建议

* **越小越好**：只放 `rollout` 真正需要的字段，便于序列化/记录/回放。
* **可序列化**：用 `dataclass` 或 Pydantic `BaseModel`（方便校验与保存）。
* **明确评测接口**：如果要自动打分，Scenario 里应包含参考答案或可据此得分的线索。
* **与数据生成对齐**：你若用 `scenario_generator.py` 自动生成任务，它产出形如 `{"task": "...", "difficulty": ...}` 的 JSON；训练脚本再把它转成 `McpScenario`（见 `mcp_rl/train.py` 118–137 行）。


# model.get_step()
get_step() 用来拿“这个可训练模型目前处在第几步（global training step）”。它返回一个整数，比如 0、1、2……，表示你已经完成并落盘的最新训练步，从而让训练/评测/保存都能接着上次的进度继续，而不是从头来。
model.get_step()（src/art/model.py）是个 async 方法 → 调用后端的 backend._get_step(model)。
本地后端里（src/art/local/backend.py）实际实现为：
若是 TrainableModel，就从模型的输出目录里找最新的 checkpoint 目录，路径约为
.../<project>/models/<name>/checkpoints/<step:04d>
并取其中最大的 <step> 作为当前 step（见 src/art/utils/get_model_step.py）。
如果没有找到，则返回 0。注册后会有 0000 这个初始 checkpoint。
之所以是 async，是因为也可能通过 REST 接口查询远端/本地服务（见 src/art/backend.py 与 src/art/cli.py）。
