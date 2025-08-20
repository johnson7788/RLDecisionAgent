# 启动LLM代理
cd ART_mcp-rl
python LLM_cache.py

# 生成不同模型的玩2048的结果
[generate_benchmarks.py](generate_benchmarks.py)

# 绘制这些模型的结果到图像
[display_benchmarks.py](display_benchmarks.py)