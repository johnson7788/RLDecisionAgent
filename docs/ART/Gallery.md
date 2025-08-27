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


#  FastLanguageModel 参数
下面把你贴的 `FastLanguageModel.from_pretrained(...)` / `FastModel.from_pretrained(...)` 里常见参数逐个用中文说明，并给出“什么时候用/怎么选”的简要建议。括号里是默认值（来自你给的源码）。

# FastLanguageModel.from\_pretrained(...)（纯文本/常见 LLM）

* **model\_name** (`"unsloth/Llama-3.2-1B-Instruct"`): 要加载的模型或路径。既可指向基础模型，也可指向只含 LoRA 的仓库；如果检测到是 LoRA 适配器，会自动回溯并加载其 `base_model` 再套上适配器。
  用法：传 HF Hub 名称或本地目录。若仓库包含 `adapter_config.json` 就视为 PEFT 适配器。([Hugging Face][1])
* **max\_seq\_length** (`2048`): 生成/训练时支持的最大上下文长度（Unsloth 会据此创建/补丁缓存）。
  经验：设为你准备训练/推理所需的最大值。
* **dtype** (`None`): `torch.float16` / `torch.bfloat16`（或留空自动选 bfloat16 支持则用 bf16，否则 fp16）。
  经验：A100/H100/4090 等优先 `bfloat16`；老显卡用 `float16`。Unsloth 文档也建议按硬件自动选择。([Unsloth 文档][2])
* **load\_in\_4bit** (`True`): 是否以 **bitsandbytes 4-bit** 量化加载（QLoRA 常用，显存友好）。Unsloth 会把 `quantization_config`（NF4 + double quant）写进 `model.config`。
  何时用：显存紧/准备做 QLoRA/LoRA；推理或微调都可。([Hugging Face][3])
* **load\_in\_8bit** (`False`): 以 8-bit 量化加载（LLM.int8）。与 `load_in_4bit` 互斥。([Hugging Face][4])
* **full\_finetuning** (`False`): 是否进行“全参微调”。若开了它，会强制关闭 4/8bit（源码里直接改为浮点权重训练）。
  何时用：你真的要全参训练且显存够用。
* **token** (`None`): Hugging Face 访问令牌，用于拉取私有模型/推送模型等。文档在保存/导出章节多次强调需设置 token。([Unsloth 文档][5])
* **device\_map** (`"sequential"`): 模型分布到设备的方式。常见还有 `"auto"`, `"balanced"`, `"balanced_low_0"` 等（由 Accelerate 计算）。
  何时用：单卡默认即可；多卡/大模型建议 `"auto"`。([Hugging Face][6])
* **rope\_scaling** (`None`): RoPE 缩放配置（如扩上下文时的 YaRN/LLAMA3 方案，取决于底模支持）。
  备注：是否支持由 transformers 版本/底模决定。([Hugging Face][7])
* **fix\_tokenizer** (`True`): 针对部分本地权重，优先用同目录下的 tokenizer 三件套（防止被底模覆盖）。
  何时用：本地自带 tokenizer（源码已做存在性检查）。
* **trust\_remote\_code** (`False`): 允许执行模型仓库自定义代码（某些模型必需）。([Hugging Face][8])
* **use\_gradient\_checkpointing** (`"unsloth"`): 启用 **Unsloth 定制的梯度检查点**，更省显存；文档/示例里称可显著降低 VRAM、适合长上下文训练。可设为 `True` 或 `"unsloth"`（推荐后者）。([Unsloth 文档][9])
* **resize\_model\_vocab** (`None`): 需要时调整词表大小（会调用 `resize_token_embeddings`）。
* **revision** (`None`): 指定 Hub 仓库的分支/tag/commit。([Hugging Face][8])
* **use\_exact\_model\_name** (`False`): 关闭 Unsloth 的“名称重写/动态量化名”逻辑，按字面 `model_name` 加载（例如你已下载到本地时）。文档在“环境标志/疑难排解”里也展示过“强制精确名”的场景。([Unsloth 文档][10])
* **fast\_inference** (`False`): 走 **vLLM** 推理路径（需 `pip install vllm`）；Unsloth 文档有专门的 vLLM 保存/部署页面。
  何时用：部署推理服务、需要吞吐/并发更高。([Unsloth 文档][11])
* **gpu\_memory\_utilization** (`0.5`): vLLM 显存利用率（仅 `fast_inference=True` 时有用）。
* **float8\_kv\_cache** (`False`): KV-cache 用 float8 存储以省显存（更偏推理优化）。
* **random\_state** (`3407`): 随机种子（用于训练/采样的可复现性）。
* **max\_lora\_rank** (`64`): 给 LoRA 的 rank 上限（便于后续 `get_peft_model` 时做检查/优化，一般保默认即可）。
* **disable\_log\_stats** (`True`): 关闭若干统计日志输出，减少噪声。

> 返回值：`(model, tokenizer)`；官方“推理”页也用该二元组示例，并建议再调用 `FastLanguageModel.for_inference(model)` 以开启本地 2× 推理优化。([Unsloth 文档][12])

---

# FastModel.from\_pretrained(...)（更通用：含多模态/视觉）

多数参数与上面一致，另有：

* **return\_logits** (`False`): 为评测等场景直接返回 logits（也可用环境变量开关）。([Unsloth 文档][10])
* **fullgraph** (`True`): 与 Unsloth 的 compile/图优化相关（控制是否允许图中断等，影响速度/稳定性）。
* **auto\_model** (`None`): 自动选 `AutoModelForCausalLM`（文本）或 `AutoModelForVision2Seq`（VLM）；源码会根据 `architectures/vision_config` 判断是否是多模态。
* **whisper\_language / whisper\_task**: 若加载 Whisper 系列，指定 ASR 语言/任务。
* **unsloth\_force\_compile** (`False`): 强制编译（debug/性能调优用）。
* 视觉/多模态相关的教程在“Vision Fine-tuning”和各模型专项页（Pixtral、Qwen2-VL、Llama 3.2 Vision 等）。([Unsloth 文档][13], [CSDN博客][14])

---

## 常见用法小抄

```python
# 1) QLoRA 微调（4-bit 省显存，推荐）
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8b-unsloth-bnb-4bit",
    load_in_4bit=True, dtype=None, use_gradient_checkpointing="unsloth"
)
# 接着：FastLanguageModel.get_peft_model(...) 开 LoRA
```

（QLoRA + 4bit 的背景与优点详见 HF 官方量化与 PEFT 文档。）([Hugging Face][15])

```python
# 2) 全参微调（需要大显存）
model, tok = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B",
    full_finetuning=True, load_in_4bit=False, load_in_8bit=False,
    dtype=torch.bfloat16, device_map="auto"
)
```

（`device_map="auto"` 由 Accelerate 自动做“大模型切分”以适配多卡/单机显存。）([Hugging Face][6])

```python
# 3) vLLM 快速推理
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    fast_inference=True, gpu_memory_utilization=0.8
)
```

（Unsloth 支持与 vLLM 的保存/部署联动。）([Unsloth 文档][11])

---

## 重要行为/坑点（来自源码）

* **LoRA 适配器与基座不能同仓**：若同一目录同时存在 `config.json` 和 `adapter_config.json`，函数会报错，要求分仓（一个放基座，一个放 LoRA）。这是为了避免“到底加载谁”的二义性。
* **自动量化配置**：`load_in_4bit=True` 时，函数会把 `nf4 + double quant` 等写入 `model.config.quantization_config`，便于后续在 transformers 框架下正确识别量化状态。相关量化原理参见 HF 官方文档。([Hugging Face][3])
* **Unsloth 梯度检查点**：`"unsloth"` 方案更省显存、适合长上下文任务；官方示例在 RL/DPO 教程里也默认这么设。([Unsloth 文档][9])
* **token 登录**：私有模型/推送到 Hub 时需要 HF token。([Unsloth 文档][5])

如果你给我你现在的**显卡型号/显存**和**目的（微调/推理/多模态？）**，

[1]: https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model?utm_source=chatgpt.com "Models - Hugging Face"
[2]: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide?utm_source=chatgpt.com "Fine-tuning LLMs Guide | Unsloth Documentation"
[3]: https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes?utm_source=chatgpt.com "Bitsandbytes"
[4]: https://huggingface.co/docs/transformers/v4.27.0/main_classes/quantization?utm_source=chatgpt.com "Quantize Transformers models - Hugging Face"
[5]: https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-ollama?utm_source=chatgpt.com "Saving to Ollama | Unsloth Documentation"
[6]: https://huggingface.co/docs/accelerate/usage_guides/big_modeling?utm_source=chatgpt.com "Big Model Inference - Hugging Face"
[7]: https://huggingface.co/docs/transformers/main/index?utm_source=chatgpt.com "Transformers - Hugging Face"
[8]: https://huggingface.co/docs/transformers/main_classes/model?utm_source=chatgpt.com "Models - Hugging Face"
[9]: https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide/reinforcement-learning-dpo-orpo-and-kto?utm_source=chatgpt.com "Reinforcement Learning - DPO, ORPO & KTO - Unsloth"
[10]: https://docs.unsloth.ai/basics/troubleshooting-and-faqs/unsloth-environment-flags?utm_source=chatgpt.com "Unsloth Environment Flags | Unsloth Documentation"
[11]: https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm?utm_source=chatgpt.com "Saving to VLLM | Unsloth Documentation"
[12]: https://docs.unsloth.ai/basics/running-and-saving-models/inference "Inference | Unsloth Documentation"
[13]: https://docs.unsloth.ai/basics/vision-fine-tuning?utm_source=chatgpt.com "Vision Fine-tuning | Unsloth Documentation"
[14]: https://blog.csdn.net/raozhongbo/article/details/149329645?utm_source=chatgpt.com "使用unsloth模型微调过程_unsloth微调全流程-CSDN博客"
[15]: https://huggingface.co/docs/transformers/v4.48.0/en/quantization/bitsandbytes?utm_source=chatgpt.com "bitsandbytes"


# 训练模型的保存结果
解释下面的ART模型训练的保存的信息
tree /workspace/verl/ART/.art/hn_title_generation/
/workspace/verl/ART/.art/hn_title_generation/
└── models
    └── 001
        ├── checkpoints
        │   ├── 0251
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0252
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0253
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0254
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0255
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0256
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0257
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0258
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0259
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0260
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0261
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0262
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0263
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0264
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0265
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0266
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0267
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0268
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0269
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0270
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0271
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0272
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0273
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0274
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0275
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0276
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0277
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   ├── 0278
        │   │   ├── README.md
        │   │   ├── adapter_config.json
        │   │   ├── adapter_model.safetensors
        │   │   ├── added_tokens.json
        │   │   ├── chat_template.jinja
        │   │   ├── merges.txt
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   ├── training_args.bin
        │   │   └── vocab.json
        │   └── 0279
        │       ├── README.md
        │       ├── adapter_config.json
        │       ├── adapter_model.safetensors
        │       ├── added_tokens.json
        │       ├── chat_template.jinja
        │       ├── merges.txt
        │       ├── special_tokens_map.json
        │       ├── tokenizer.json
        │       ├── tokenizer_config.json
        │       ├── training_args.bin
        │       └── vocab.json
        ├── history.jsonl
        ├── logs
        │   └── vllm.log
        ├── model.json
        ├── tensors
        │   ├── advantages.pt
        │   ├── assistant_mask.pt
        │   ├── group_ids.pt
        │   ├── input_pos.pt
        │   ├── logprobs.pt
        │   ├── parent_ids.pt
        │   ├── tokens.pt
        │   └── weights.pt
        └── trajectories
            ├── train
            │   ├── 0000.jsonl
            │   ├── 0001.jsonl

            │   ├── 0273.jsonl
            │   ├── 0274.jsonl
            │   ├── 0275.jsonl
            │   ├── 0276.jsonl
            │   ├── 0277.jsonl
            │   └── 0278.jsonl
            └── val
                ├── 0051.jsonl
                ├── 0101.jsonl
                ├── 0151.jsonl
                ├── 0201.jsonl
                └── 0251.jsonl

37 directories, 614 files

推理 / 继续训练 / 导出合并

# 顶层与运行编号

* `.art/hn_title_generation/models/001/`
  `001`通常是这次实验 / 运行(run)的编号。一个工程里可以有 `002、003…` 表示不同次训练。

* `model.json`
  记录本次运行的关键信息（例如基座模型名称、适配器类型/路径、超参摘要等）。用来让框架在加载/恢复时知道“我是谁、基于谁”。

* `history.jsonl`
  按 step 追加的训练日志（JSON Lines）。里面通常包含训练/评估指标（如loss、reward、学习率、步数等）。用它可以画曲线或定位“最佳 step”。

* `logs/vllm.log`
  推断/采样阶段使用 vLLM 时的日志（端口、吞吐、错误堆栈等）。

# checkpoints：每个保存点一套可复现工件

`checkpoints/0251 ~ 0279/` 这些编号基本就是**全局 step（左侧零填充）**。每个子目录都能独立用于推理或恢复训练，内含：

* `adapter_model.safetensors` + `adapter_config.json`
  LoRA/QLoRA 等 **PEFT 适配器**权重与配置（如 r、alpha、target\_modules…）。这是“这次训练学到的东西”的主体。
* `training_args.bin`
  🤗Transformers 的 `TrainingArguments` 对象序列化，用于**恢复训练超参**。
* `tokenizer.json / tokenizer_config.json / vocab.json / merges.txt / special_tokens_map.json / added_tokens.json`
  **分词器**全套文件，保证推理/继续训练时词表与训练时一致。
* `chat_template.jinja`
  聊天模板（系统/用户/助手消息如何拼成模型输入）。推理由它确保**提示词格式**与训练一致。
* `README.md`
  该 checkpoint 的简要说明（例如创建时间、度量指标摘要）；不同版本内容可能略有差异。

> 选“最佳”checkpoint：
> 一般参考 `history.jsonl`（看验证集指标）或各 checkpoint 自带的 `README.md` 摘要。若未显式保存“best”，通常取**最后一个**（这里是 `0279`）做推理的起点，然后再按指标回退对比。

# tensors：中间张量缓存（非模型权重）

```
tensors/
├── advantages.pt       # PPO/GRPO 一类算法里计算得到的优势 A_t
├── assistant_mask.pt   # 掩码：哪些 token 属于“assistant”回复段
├── group_ids.pt        # 样本分组/采样批次的映射
├── input_pos.pt        # 位置索引（重构序列时用）
├── logprobs.pt         # 采样得到的对数概率
├── parent_ids.pt       # 回溯到原始轨迹/样本的索引
├── tokens.pt           # 实际 token 序列缓存
└── weights.pt          # 训练用权重（如重要性采样权重等）
```

这些是**加速继续训练/复现对齐计算**的缓存；**不是**可直接用于推理的模型权重。丢失不会影响用 checkpoint 推理，但会影响“无缝恢复”某些 RL 训练阶段。

# trajectories：训练/验证“轨迹”数据

```
trajectories/
├── train/0000.jsonl ~ 0278.jsonl
└── val/0051.jsonl, 0101.jsonl, ...
```

* 每个 `.jsonl` 是一个**分片**（shard），行级 JSON 记录一次“对话→模型输出→打分/反馈”的**轨迹**。
* 常见字段（不同版本会有差别）：`messages`（或 `prompt/completion`）、`response`、`reward(s)`、`scores`、`meta`（温度、采样参数、基座模型哈希、时间戳等）、以及可选的 `logprobs`。
* `val/` 是验证集轨迹，训练过程用于周期性评估。

# 常见操作速查

## 1) 用某个 checkpoint 做推理（加载 LoRA 适配器）

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "<你的基座模型路径或HF模型名>"   # 通常可在 models/001/model.json 找到
ckpt_dir = "/workspace/verl/ART/.art/hn_title_generation/models/001/checkpoints/0279"

tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, ckpt_dir)   # 加载 adapter_model.safetensors

# 如果有 chat_template.jinja，建议按模板构造输入
text = tok.apply_chat_template(
    [{"role":"user","content":"帮我为这篇文章写一个Hacker News风格标题：..."}],
    tokenize=False, add_generation_prompt=True, chat_template=open(f"{ckpt_dir}/chat_template.jinja").read()
)

inputs = tok(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(outputs[0], skip_special_tokens=True))
```

## 2) 把 LoRA 合并成“完整权重”（便于独立部署）

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto")
model = PeftModel.from_pretrained(base, ckpt_dir)
merged = model.merge_and_unload()                   # 合并 LoRA
merged.save_pretrained("/path/to/merged_model")
tok.save_pretrained("/path/to/merged_model")
```

> 合并后就不再依赖 `adapter_model.safetensors`，部署更简单；但会占用更多磁盘。

## 3) 从某个 checkpoint 恢复继续训练

大多数基于 🤗Trainer/TRL 的脚本支持：

```bash
python train.py \
  --resume_from_checkpoint /workspace/verl/ART/.art/hn_title_generation/models/001/checkpoints/0270
```

它会读取 `training_args.bin` 和该目录内分词器/模板，继续按相同设置训练。

# 清理与归档建议

* **长期保留**：`checkpoints/（最佳或最后一个）`、`model.json`、`history.jsonl`、`chat_template.jinja`。
* **按需保留**：`trajectories/`（为了论文复现/审计很重要；但体积大）。
* **可清理**：早期大量 checkpoint 与 `tensors/`（若不再需要从中间 step 精确恢复）。
* **日志**：`logs/vllm.log` 留作排错，过旧可压缩归档。

---

# Art.TrainableModel 和Art.Model