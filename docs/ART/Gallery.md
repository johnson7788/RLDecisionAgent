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