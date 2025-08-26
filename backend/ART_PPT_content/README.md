# 使用强化学习训练模型生成PPT的内容

## 安装依赖
pip install -r requirements.txt

## 数据集
使用智谱的web搜索,添加搜索缓存
https://docs.bigmodel.cn/cn/guide/tools/web-search#mcp-server

## 生成问题场景，覆盖现有场景
python generate_questions.py
python data_convert.py
输出文件,训练数据和测试数据
[questions.jsonl](questions.jsonl)
[scenarios.jsonl](scenarios.jsonl)


## MCP工具准备，搜索工具
[mcp_search](mcp_search)


## 训练模型

## 修改prompt
[prompt.py](train_test_model%2Fprompt.py)

## 训练模型
[train_test_model](train_test_model)
python -m train_test_model.model_train

