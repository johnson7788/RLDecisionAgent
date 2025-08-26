# 使用强化学习训练模型生成PPT的内容

## 安装依赖
pip install -r requirements.txt

## 数据集
## MCP工具准备，搜索工具
[mcp_search](mcp_search)
使用智谱的web搜索,添加搜索缓存
https://docs.bigmodel.cn/cn/guide/tools/web-search#mcp-server

## 问题场景
[scenarios](train_test_model%2Fmcp_search%2Fscenarios)


## 训练模型

## 修改prompt
[prompt.py](train_test_model%2Fprompt.py)

## 训练模型
[train_test_model](train_test_model)
python -m train_test_model.model_train

