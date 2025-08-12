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
## Step1 数据准备
```
cd ART/examples/mcp-rl
创建.env文件
# cat .env
OPENAI_API_KEY=sk-proj-x-xxxx
ALPHAVANTAGE_API_KEY=SGxxxx
WANDB_API_KEY=c93xxx
BALLDONTLIE_API_KEY=83xxx
# 复制一份到mcp_rl中
cp .env mcp_rl 

准备数据(其实已经存在数据了，在每个servers/xxx/scenarios/train.jsonl和val.jsonl)
python -m mcp_rl.scenario_generator servers/python/mcp_alphavantage/server_params.py
输出
weave: Please login to Weights & Biases (https://wandb.ai/) to continue...
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
wandb: No netrc file found, creating one.
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Currently logged in as: johnson- to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
weave: wandb version 0.21.1 is available!  To upgrade, please run:
weave:  $ pip install wandb --upgrade
weave: Logged in as Weights & Biases user: johnson-.
weave: View Weave data at https://wandb.ai/johnson-/mcp-agent-training/weave
Generating 24 scenarios using servers/python/mcp_alphavantage/server_params.py...
Output will be saved to: servers/python/mcp_alphavantage/scenarios/
Saved 16 training scenarios to servers/python/mcp_alphavantage/scenarios/train.jsonl
Saved 8 validation scenarios to servers/python/mcp_alphavantage/scenarios/val.jsonl

Generated 24 scenarios:
1. Task: Perform a comprehensive analysis by retrieving IBM's daily time series data, calculating both the SMA and RSI, then creating an in-depth report with a complete summary and analysis of the stock's trends.
   Difficulty: 5/5
2. Task: Calculate the 30-day Simple Moving Average (SMA) for Amazon (AMZN), compare it with the current stock price, and produce a thorough analysis and summary report on market trends.
   Difficulty: 2/5
3. Task: Search for companies with 'Google' in their name, retrieve related symbols, and produce a comprehensive report including a summary of company overviews and potential investment insights.
   Difficulty: 3/5
4. Task: Get the real-time stock quote for Nvidia (NVDA) and create a brief analysis report including a summary of its current market performance.
   Difficulty: 1/5
5. Task: Calculate the 14-day RSI for Alphabet (GOOG) on a daily interval, then generate a comprehensive analysis report that includes a summary of the market conditions and trends.
   Difficulty: 2/5
6. Task: Retrieve the weekly RSI for AMD, compile the indicator results, and develop a detailed trend analysis report that includes a comprehensive summary and market recommendations.
   Difficulty: 3/5
7. Task: Get the current stock price for Microsoft (MSFT) and generate a summary of the work done along with a thorough analysis and report of the real-time data.
   Difficulty: 1/5
8. Task: Perform a symbol search using the keyword 'energy' to identify potential investment opportunities in the energy sector, and produce a comprehensive report with analysis and summary of the identified stocks.
   Difficulty: 3/5
9. Task: Fetch the real-time stock quote for Tesla (TSLA) and compare it with its 30-day SMA to generate a detailed report with a summary and analysis of potential buy/sell signals.
   Difficulty: 3/5
10. Task: Retrieve daily time series data for Apple (AAPL) for the last 100 days, analyze the price trends, and generate a detailed summary and analysis report.
   Difficulty: 2/5
11. Task: Obtain the company overview for IBM and generate a report that includes a detailed summary and analysis of the company’s fundamental data.
   Difficulty: 1/5
12. Task: Perform a symbol search for companies in the pharmaceutical sector using the keyword 'pharma', then generate a report summarizing potential investment opportunities with detailed analysis and summary.
   Difficulty: 3/5
13. Task: Calculate the 14-day Relative Strength Index (RSI) for Netflix (NFLX) on a daily interval, then compile a comprehensive analysis report with a summary of the technical indicators.
   Difficulty: 2/5
14. Task: Calculate the SMA for Cisco (CSCO) using daily data, compare it with recent price movements, and generate a report with a thorough summary and technical analysis.
   Difficulty: 2/5
15. Task: Fetch the company overview for Facebook (META) and generate a detailed report that includes a thorough summary and analysis of the company’s fundamentals.
   Difficulty: 1/5
16. Task: Retrieve the company overview for McDonald's (MCD) and compile a report that includes a detailed summary and analysis of its fundamentals and market performance.
   Difficulty: 2/5
17. Task: Search for stock symbols using the keyword 'bank' to identify potential financial investments, and generate a detailed summary report that includes an analysis of each candidate.
   Difficulty: 3/5
18. Task: Get a real-time stock quote for Boeing (BA), retrieve its daily time series data for context, and develop an analysis report including a detailed summary of historical and current performance.
   Difficulty: 4/5
19. Task: Fetch the company overview for Twitter (TWTR) and generate a comprehensive report that includes a detailed summary and analysis of its fundamental health and market positioning.
   Difficulty: 1/5
20. Task: Fetch the real-time stock quote for Goldman Sachs (GS), calculate the 14-day RSI, and develop an integrated market analysis report with a thorough summary and recommendations.
   Difficulty: 4/5
21. Task: Search for companies in the retail sector, select one, retrieve its company overview and daily time series data, and compile a detailed analytical report with a thorough summary of its market outlook.
   Difficulty: 5/5
22. Task: Retrieve daily time series data for Intel (INTC), analyze historical price movements, and produce a detailed report including a summary of trends and insights.
   Difficulty: 3/5
23. Task: Gather daily time series data for Salesforce (CRM), compute the 30-day SMA, and prepare a comparative analysis report that includes a detailed summary of tech sector trends.
   Difficulty: 4/5
24. Task: Retrieve daily time series data for Oracle (ORCL) for the recent trading period, analyze the price movements and trends, and produce a comprehensive summary report.
   Difficulty: 3/5
```

## Step2 开始训练
```
cd ART/examples/mcp-rl
export CUDA_VISIBLE_DEVICES=1
python docs/ART/load_model.py
pip install polars torchtune trl unsloth # 安装一个依赖包
# 取消上传试验结果到s3
│ ✔  Edit examples/mcp-rl/mcp_rl/train.py:         await backend._experim... =>         # await backend._exper...        │
 │                                                                                                                        │
 │    170           print("starting train")                                                                               │
 │    171           await model.train(groups, config=art.TrainConfig(learning_rate=learning_rate))                        │
 │    172                                                                                                                 │
 │    173 -         await backend._experimental_push_to_s3(                                                               │
 │    174 -             model,                                                                                            │
 │    175 -         )                                                                                                     │
 │    173 +         # await backend._experimental_push_to_s3(                                                             │
 │    174 +         #     model,                                                                                          │
 │    175 +         # )                                                                                                   │
 │    176                                                                                                                 │
 │    177                                                                                                                 │
 │    178   def main():

python -m mcp_rl.train --models=mcp-14b-alpha-001
```