# 示例
https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb#scrollTo=OsrwCDQ5cviC

# colab的显卡
Tesla T4 16GB

# 安装
!uv pip install -q openpipe-art==0.3.11.post5 langchain-core tenacity "mcp>=1.11.0" "gql<4" aiohttp --no-cache-dir

# 配置smithery
https://smithery.ai/
配置
OPENROUTER_API_KEY = "sk-or-v1-xxx" 

打开https://smithery.ai/server/exa， 然后点击点击 Get URL with keys instead, 是网络搜索的工具
SMITHERY_MCP_URL = "https://server.smithery.ai/exa/mcp?api_key=552ddb78-0e95-4998-be87-b936502a5a97&profile=stiff-sole-4FpnEv"

# 运行
python mcp_rl.py