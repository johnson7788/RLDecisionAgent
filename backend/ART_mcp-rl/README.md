# 文件信息
   * all_experiments.py: 定义了多个用于训练的机器学习模型配置。它使用 pydantic 和一个名为 art 的自定义库来设置不同模型（如
     mcp-7b-001, mcp-14b-001）的参数，包括基础模型、学习率、训练周期等。这些模型似乎是为与
     MCP（模型上下文协议）服务器交互而设计的。
   * pyproject.toml: 项目的配置文件，使用 uv 作为包管理器。它定义了项目名称 (art-mcp)、依赖项（如 aiohttp, mcp, openai,
     openpipe-art 等）以及开发依赖项。
   * README.md: 项目的说明文件，可能包含项目的目标、设置说明和使用方法。（注意：文件内容为空）。
   * run_remote.py: 一个使用 sky (SkyPilot) 库在远程计算集群上运行训练任务的脚本。它负责设置远程环境、同步代码并执行指定的 Python
     模块（如训练或基准测试脚本）。

# 训练好的本地模型的验证程序
[mcp_rl_test.py](mcp_rl_test.py)


  mcp_rl/ 目录 (核心 RL 逻辑)

   * __init__.py: 将 mcp_rl 目录标记为 Python 包，并导出了关键的类和函数，如 rollout, McpScenario, McpServer。
   * checks.py: 包含一个使用 LLM（如 GPT-4）来评估任务是否成功完成的函数
     check_successful。它通过分析代理与环境的交互历史来判断任务目标是否达成。
   * mcp_server.py: 定义了与 MCP 服务器交互的抽象基类 McpServer 和具体的实现，如 LocalMcpServer（本地 stdio 服务器）和
     RemoteMcpServer（远程 HTTP 服务器）。这为训练和评估提供了灵活的服务器连接方式。
   * rollout.py: 实现了核心的 rollout 函数，该函数模拟一个 AI 代理（由 art.Model 定义）在给定的 McpScenario（场景）中与 MCP
     服务器的交互过程。它负责处理多轮对话、工具调用和任务完成的逻辑。
   * scenario_generator.py: 一个用于自动生成训练和验证场景的脚本。它连接到指定的 MCP
     服务器，获取其工具列表，然后使用大型语言模型（如 OpenAI 的 o3-mini）创建多样化的任务，并将其保存为 .jsonl 文件。
   * train.py: 项目的训练脚本。它加载模型配置、训练数据（场景），并使用 rollout 函数收集经验轨迹。然后，它使用 ruler_score_group
     评估这些轨迹的质量，并对模型进行微调。
   * utils.py: 提供辅助函数，例如 get_content_text，用于从 MCP 工具调用的结果中提取文本内容。

  mcp_rl/benchmarks/ 目录 (基准测试)

   * generate_benchmarks.py: 运行基准测试的脚本。它加载预定义的验证场景，针对多个不同的模型（包括训练的模型和基准模型如
     GPT-4o）运行 rollout，并使用 RULER 进行比较评估，最后记录结果。

  servers/ 目录 (MCP 服务器实现)

  这个目录包含了多个具体的 MCP 服务器实现，每个服务器都提供了与特定 API（如 Alpha Vantage、Balldontlie、Google
  Maps）交互的工具。

   * `python/mcp_alphavantage/`:
       * server.py: 实现了与 Alpha Vantage API 交互的 MCP 服务器。它定义了如 get_stock_quote, get_time_series_daily 等工具。
       * server_params.py: 为该服务器定义了 StdioServerParameters，用于在本地启动服务器进程。
       * scenarios/ 和 scenarios.jsonl: 包含了为 Alpha Vantage 服务器生成的训练和验证任务。

   * `python/mcp_balldontlie/`:
       * server.py: 实现了与 Balldontlie NBA 统计 API 交互的 MCP 服务器，提供了查询球队、球员和比赛信息的工具。
       * server_params.py: 为该服务器定义了启动参数。
       * scenarios/: 包含了为 Balldontlie 服务器生成的训练和验证任务。

   * `python/mcp_googlemaps/`:
       * server.py: 实现了与 Google Maps API 交互的 MCP 服务器，提供了地理编码、地点搜索等工具。
       * server_params.py: 为该服务器定义了启动参数。
       * pyproject.toml: 该特定服务器的 Python 项目配置文件。


