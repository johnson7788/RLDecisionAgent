from fastmcp import FastMCP

# 模拟数据
chunks = [
    "Python 是一种广泛使用的高级编程语言。",
    "FastMCP 是一个快速构建 MCP 工具的 Python 库。",
    "Faiss 是 Facebook AI 开发的相似度搜索库。",
    "SentenceTransformer 用于生成文本向量表示。",
    "MCP（Model Context Protocol）允许工具与模型交互。"
]

mcp = FastMCP(name="wiki retrieval mcp (mock)")


@mcp.tool(
    name="retrieve",
    description="模拟从 wikipedia 数据集中检索相关 chunk",
)
def retrieve(query: str) -> list:
    """
    模拟检索相关的 chunks，不使用向量搜索，仅做关键词匹配。
    """
    top_k = 3  # 返回前 3 条
    results = []

    for i, chunk in enumerate(chunks):
        score = chunk.lower().count(query.lower())  # 简单的匹配分数
        if score > 0:
            results.append({"chunk": chunk, "chunk_id": i, "score": score})

    # 按匹配次数排序
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    # 启动 MCP SSE 服务
    mcp.run(transport="sse", host="127.0.0.1", port=8099)
