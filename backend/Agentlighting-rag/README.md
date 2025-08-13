# 训练RAG Agent
[rag_agent.py](..%2F..%2Fagent-lightning%2Fexamples%2Frag%2Frag_agent.py)

# 安装
pip install agentlightning openai-agents

# 步骤
在单个机器节点上运行，需要2个GPU，每个GPU至少需要24GB内存。

1. 在wiki_retriever_mcp文件夹中准备RAG数据集。需要维基片段(`nq_list.pkl`)和Faiss索引(`nq_hnsw_faiss_n32e40.index`)。
2. 在data文件夹中需要有`musique_train.parquet`和`musique_dev_128.parquet数据`。
3. 为wiki检索器MCP设置环境：`bash wiki_retriever_install.sh`。这将安装所需的包并为wiki检索器MCP设置环境。
4. 启动wiki检索器MCP：`python wiki_retriever_mcp.py`。这将启动wiki检索器MCP服务器。
5. export CUDA_VISIBLE_DEVICES=1,2
5. 启动Ray：bash scripts/restart_ray.sh。要使用Wandb，需要在启动Ray之前设置WANDB_API_KEY环境变量。
6. 运行代理：`python rag_agent.py`。这将默认自动启动4个代理工作进程。
7. 在另一个终端中，启动训练服务器：`bash train.sh`。

## 评估

