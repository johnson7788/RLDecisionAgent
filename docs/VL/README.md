# 安装
# 环境部署

##  尝试使用Areal的镜像
```
docker pull ghcr.io/inclusionai/areal-runtime:v0.3.0.post2
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name areal ghcr.io/inclusionai/areal-runtime:v0.3.0.post2 sleep infinity
docker start areal
docker exec -it areal bash
```

## ~/.bashrc中配置使用的GPU和实用的hugging face镜像
CUDA_VISIBLE_DEVICES=1
HF_ENDPOINT=https://hf-mirror.com

## 设置pip镜像源
```
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
# 克隆强化学习训练框架
```
git clone https://github.com/OpenPipe/ART.git
cd ART
pip install .
pip install ".[backend]"
```

## 设置代理，安装git上的项目
```
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPs_PROXY=http://127.0.0.1:7890
pip install 'torchtune @ git+https://github.com/pytorch/torchtune.git'
pip install 'unsloth-zoo @ git+https://github.com/bradhilton/unsloth-zoo'
```

## WanDB docker 训练记录
```
docker pull wandb/local
docker run -d --restart always -v wandb:/vol -p 3005:8080 --name wandb-local wandb/local
#会提示您配置打开浏览器http://localhost:3005/authorize，新建一个本地普通用户, 粘贴key
输入邮箱和用户名创建一个本地的用户，得到类似这样的KEY， local-f2ca8cd44276ac92ca0a2c12641a6902beb6847d
粘贴到.env的环境变量配置文件中
```

# 多模态模型
对比式（Contrastive，InternVL-C）检索与生成式（Generative，InternVL-G）

| 维度     | 对比式检索（InternVL-C）                                    | 生成式检索（InternVL-G）                                        |                   |
| ------ | ---------------------------------------------------- | -------------------------------------------------------- | ----------------- |
| 基本思想   | 学到**图-文同一语义空间**，用相似度打分（类似 CLIP）。                     | 以“看图→生成文本”的**条件似然**来评估匹配度（文本作为目标序列）。                     |                   |
| 训练目标   | InfoNCE/对比损失：拉近正样本、推远负样本。                            | 最大化 p(text                                               | image) 或等价的自回归似然。 |
| 前向接口   | `mode='InternVL-C'` 返回 `[N_img, N_txt]` 的相似度 logits。 | `mode='InternVL-G'` 返回基于生成概率的匹配 logits（无需真正生成整句，内部计算似然）。 |                   |
| 打分解释   | “这张图”与“这些句子”谁更像 → softmax 后近似概率。                     | “这张图”生成“这句话”的（对数）概率有多大。                                  |                   |
| 召回/扩展性 | 适合**大规模库**：可离线提取向量，用 ANN 近似检索。                       | 不便向量索引；更适合**小候选集重排**或最终裁决。                               |                   |
| 速度/算力  | **快**：一次前向即可得到整块相似度矩阵；可批处理。                          | **慢**：需要基于解码器的序列似然计算（比对比式更重）。                            |                   |
| 语义颗粒度  | 粗到中等粒度（短语/句子层面很强）。                                   | 细粒度、一致性较好（更贴近“逐词解释这张图”）。                                 |                   |
| 对提示敏感性 | 对提示词形式较**稳健**（同义改写影响较小）。                             | 对提示设计更**敏感**（长度/措辞会影响似然）。                                |                   |
| 多语言/跨域 | 多语种泛化强，只要文本编码器覆盖语言即可。                                | 依赖解码器对该语言的建模质量，**长度偏置**更明显。                              |                   |
| 标定/可解释 | 分数是相对相似度，**易在多候选间软max对比**。                           | 分数是条件似然，**可解释为生成难度/自然度**。                                |                   |
| 典型用法   | **Stage-1 召回**（ANN/粗排），Top-K 进入下一步。                  | **Stage-2 精排/判别**；或直接用于**caption/VQA 生成**。               |                   |


下面把**对比式（Contrastive，InternVL-C）检索**与**生成式（Generative，InternVL-G）检索**放到一张表里对比，然后给出工程落地建议与常见坑。

# 总览对比

| 维度     | 对比式检索（InternVL-C）                                    | 生成式检索（InternVL-G）                                        |                   |
| ------ | ---------------------------------------------------- | -------------------------------------------------------- | ----------------- |
| 基本思想   | 学到**图-文同一语义空间**，用相似度打分（类似 CLIP）。                     | 以“看图→生成文本”的**条件似然**来评估匹配度（文本作为目标序列）。                     |                   |
| 训练目标   | InfoNCE/对比损失：拉近正样本、推远负样本。                            | 最大化 p(text                                               | image) 或等价的自回归似然。 |
| 前向接口   | `mode='InternVL-C'` 返回 `[N_img, N_txt]` 的相似度 logits。 | `mode='InternVL-G'` 返回基于生成概率的匹配 logits（无需真正生成整句，内部计算似然）。 |                   |
| 打分解释   | “这张图”与“这些句子”谁更像 → softmax 后近似概率。                     | “这张图”生成“这句话”的（对数）概率有多大。                                  |                   |
| 召回/扩展性 | 适合**大规模库**：可离线提取向量，用 ANN 近似检索。                       | 不便向量索引；更适合**小候选集重排**或最终裁决。                               |                   |
| 速度/算力  | **快**：一次前向即可得到整块相似度矩阵；可批处理。                          | **慢**：需要基于解码器的序列似然计算（比对比式更重）。                            |                   |
| 语义颗粒度  | 粗到中等粒度（短语/句子层面很强）。                                   | 细粒度、一致性较好（更贴近“逐词解释这张图”）。                                 |                   |
| 对提示敏感性 | 对提示词形式较**稳健**（同义改写影响较小）。                             | 对提示设计更**敏感**（长度/措辞会影响似然）。                                |                   |
| 多语言/跨域 | 多语种泛化强，只要文本编码器覆盖语言即可。                                | 依赖解码器对该语言的建模质量，**长度偏置**更明显。                              |                   |
| 标定/可解释 | 分数是相对相似度，**易在多候选间软max对比**。                           | 分数是条件似然，**可解释为生成难度/自然度**。                                |                   |
| 典型用法   | **Stage-1 召回**（ANN/粗排），Top-K 进入下一步。                  | **Stage-2 精排/判别**；或直接用于**caption/VQA 生成**。               |                   |

* **优先用对比式（C）**：

  * 你有**上万/上百万**候选（图库/语料库）需要高效检索。
  * 想要**低延迟**服务或边缘端部署。
  * 需要做**向量索引**、跨库去重、快速相似搜索。

* **在 Top-K 上再用生成式（G）**：

  * 对结果质量要求更高，想在**细粒度**上更稳。
  * 需要对**语言流畅性/匹配一致性**更敏感的评估。
  * K 很小（如 10～100），可以承受略高算力。

> 简单策略：**C 召回 → G 精排 →（可选）生成描述/答案**。兼顾速度与效果，工程上最常见。

# 实战建议

1. **两阶段检索模板**

```python
# Stage-1: Contrastive 召回
with torch.no_grad():
    logits_img_txt, _ = model(image=pixel_values, text=input_ids, mode='InternVL-C')
probs = logits_img_txt.softmax(dim=-1)             # [N_img, N_txt]
topk_scores, topk_idx = probs.topk(k=20, dim=-1)   # 取每张图的 Top-20 文本

# Stage-2: Generative 精排（仅对 Top-K 候选做）
subset_text_ids = input_ids[topk_idx[img_i]]        # 某张图的候选文本
with torch.no_grad():
    g_logits, _ = model(image=pixel_values[img_i:img_i+1],
                        text=subset_text_ids, mode='InternVL-G')
g_probs = g_logits.softmax(dim=-1)[0]               # 重排后概率
best = g_probs.argmax().item()
```

2. **门槛与融合**

* 可设置**最小分数阈值**：若 C 与 G 都低于阈值，返回“未命中/需要更具体描述”。
* **分数融合**：例如 `final = α·softmax(C) + (1-α)·softmax(G)` 或在对数域做加权求和，实测调 α∈\[0.3,0.7]。

3. **多语言查询**

* 保持查询语言与库内描述一致性更好；或为每条候选存多语描述，C 阶段多语并排，G 阶段选择与查询语言一致的那条做精排。

4. **性能优化**

* C 阶段可**离线编码文本/图像**向量（若模型提供公开的投影向量），用 FAISS/HNSW 做 ANN；当前示例用的是模型内部打分接口，工程化时可查该仓库是否暴露嵌入导出 API。
* G 阶段**批处理**多个候选（把 Top-K 文本拼成 batch）能显著降延迟。

# 常见坑

* **EOS 设置**：做**生成**时要 `tokenizer.add_eos_token=False`，否则提示被提前终止；做**打分/检索**时通常保留 EOS 更稳。
* **长度偏置（G）**：似然对短句/模板句可能更“友好”，可以做**长度归一化**（如按 token 数平均对数似然）。
* **Padding 影响**：确保 `attention_mask` 正确，避免把 PAD 也计入似然或注意力。
* **安全性**：`trust_remote_code=True` 只在可信仓库使用。
* **数值尺度**：C 的 logits 是相对量，比较时要在同一批次 softmax；G 的对数似然适合做**同一图像的候选间比较**，跨样本直接比较需谨慎（可做温度/长度归一化）。

# 一句话总结

* **C**：快、可扩展、擅长“先找出来”。
* **G**：细致、可解释、擅长“再判一判”。
* **最好合用**：C 粗排 + G 精排，是跨模态检索的稳妥工程范式。


# 支持性最好 Qwen2.5VL
https://huggingface.co/unsloth/models?search=qwen2.5vl
| 模型名称                                     | 参数规模    | 量化版本             | 更新时间 | 下载量   | 点赞数 |
| ---------------------------------------- | ------- | ---------------- | ---- | ----- | --- |
| Qwen2.5-VL-3B-Instruct                   | 4B      | 原始               | 5/12 | 11.5k | 3   |
| Qwen2.5-VL-3B-Instruct-bnb-4bit          | 2B      | bnb-4bit         | 5/12 | 10.3k | 4   |
| Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit  | 3B      | unsloth-bnb-4bit | 5/12 | 12.5k | 14  |
| Qwen2.5-VL-3B-Instruct-GGUF              | 3B      | GGUF             | 5/12 | 7.31k | 11  |
| **Qwen2.5-VL-7B-Instruct**               | **8B**  | 原始               | 3天前  | 30.8k | 14  |
| Qwen2.5-VL-7B-Instruct-bnb-4bit          | 5B      | bnb-4bit         | 3天前  | 40.5k | 12  |
| Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit  | 5B      | unsloth-bnb-4bit | 3天前  | 48.2k | 41  |
| Qwen2.5-VL-7B-Instruct-GGUF              | 8B      | GGUF             | 5/12 | 92.6k | 66  |
| **Qwen2.5-VL-32B-Instruct**              | **33B** | 原始               | 5/12 | 1.68k | -   |
| Qwen2.5-VL-32B-Instruct-bnb-4bit         | 18B     | bnb-4bit         | 5/12 | 2.38k | 3   |
| Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit | 20B     | unsloth-bnb-4bit | 5/12 | 4.79k | 14  |
| Qwen2.5-VL-32B-Instruct-GGUF             | 33B     | GGUF             | 5/12 | 9.66k | 5   |
| **Qwen2.5-VL-72B-Instruct**              | **73B** | 原始               | 5/12 | 211   | 1   |
| Qwen2.5-VL-72B-Instruct-bnb-4bit         | 39B     | bnb-4bit         | 5/12 | 1.37k | 15  |
| Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit | 40B     | unsloth-bnb-4bit | 5/12 | 2.38k | 7   |
| Qwen2.5-VL-72B-Instruct-GGUF             | 73B     | GGUF             | 5/18 | 3.49k | 6   |


# GRPO训练 VL
https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_5_7B_VL_GRPO.ipynb


# GRPO（Group Relative Policy Optimization）和 GSPO（Group Sequence Policy Optimization）

GRPO：群组相对策略优化算法。组内比较多个输出（responses）来估计相对优势 (relative advantage)，通常以 token 级（每个 token 或 token‐level 的似然／概率比率）来做策略更新。它是 DeepSeek （例如 DeepSeek-R1 等）系列中用于 RL from human feedback / RL 微调的一种方法。
GSPO：序列级群组策略优化（Group Sequence Policy Optimization）。由 Qwen / 通义实验室提出，作为对 GRPO 在规模、稳定性、效率上的一些不足的改进或替代。它把很多「token 级」的机制上移到「完整响应／输出序列 (sequence)」级别。

| 比较维度                                            | GRPO                                                                                                                                                                                          | GSPO                                                                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **优化单位**                                        | Token 级别：GRPO 对每个 token 的概率比（importance ratio）或者似然计算，以及每个 token 的奖励／优势参与优化。 ([知乎专栏][1])                                                                                                       | 序列级别：GSPO 定义重要性比率（ratio）基于整个输出序列（sequence likelihood），并在序列级别做裁剪（clipping）、奖励以及优化。 ([Qwen][2])                                     |
| **裁剪 (Clipping)**                               | 在 token‐级别进行裁剪／限制，以防止某些 token 导致过度更新或高方差。 ([腾讯新闻][3])                                                                                                                                         | 在 sequence 级别做裁剪，也就是说裁剪整个响应序列的比率，这样可减少因为某个 token 的极端比率带来的波动。 ([Qwen][2])                                                          |
| **奖励 (Reward)／优势 (Advantage) 的估计方式**            | 通常对一个 prompt，采样多个输出 responses，在这些组内 (group) 比较，然后计算 token‐或部分 token 的优势，相比组内平均奖励或标准差做归一化。这个平均可以是组内所有输出在每个 token 上的奖励或也可能是结果奖励 + 中间／过程奖励混合。 ([知乎专栏][1])                                        | 同样也采样多个输出响应组成组，计算它们的序列奖励／优势，但整个序列视为一个整体。这使得估计奖励信号更一致，也更能匹配我们关心的任务最终输出质量。还包括序列长度归一化来降低因输出长度不同引起的方差问题。 ([Qwen][2])                  |
| **对混合专家模型 (MoE, Mixture‐of‐Experts) 的影响 / 稳定性** | 在 MoE 情况下，因为每个 token 的激活 /专家选取 (expert activation) 随机性／变化性很大，token‐级重要性比率可能剧烈波动，导致训练不稳定。有时为了使 GRPO 在 MoE 情况下收敛，需要加一些额外策略，比如 “Routing Replay” 来缓存一些旧策略中激活的专家路径，用来在比率计算中“回放”这些路由模式。 ([Qwen][2]) | GSPO 的设计正是为了解决这类不稳定性问题的。因为它对 token 级的详细变化不敏感，而是基于整个序列，因此在 MoE 模型中可以更稳定地训练，不依赖 Routing Replay 这样的额外机制。 ([Qwen][2])                 |
| **训练效率和规模化／可扩展性**                               | 在一般规模 & 任务中表现良好；但随着输出序列变长、模型越大、MoE 结构复杂、训练计算开销增大时，其 Token‐级别的重要性比率、token 裁剪与奖励计算带来的方差和开销，会成为瓶颈。也可能训练期间「崩塌」（模型行为不稳定）或性能提升乏力。 ([腾讯新闻][4])                                                       | GSPO 在实验中表现出更好的训练效率，在相同计算资源下常常取得更高性能；并随着计算资源／模型规模的增加，性能有更好的持续提升（scaling）。此外，它能减少基础设施依赖（例如减少显存、通信、Routing Replay 等开销）。 ([Qwen][2]) |
| **对奖励信号噪声 & 方差的控制**                             | 因为 token‐级别的比率会受到某些 token 极端概率变化的影响（例如某些 token 很可能，某些 token 很少），所以 token‐级优化的方差比较大，训练不稳定性来自这部分。 ([Qwen][2])                                                                                   | GSPO 通过在序列级别做归一化，以及裁剪整个序列比率／奖励等方式，降低噪声／方差，并更好与我们关心的序列级任务（如完整回答的正确性／质量）对齐。 ([Qwen][2])                                             |

| 算法       | 优点                                                                                                                                                      | 缺点／局限性                                                                                                                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GRPO** | - 对组内多个候选输出分别比较，有利于探索多种输出，鼓励多样性。 <br> - token‐级别优化可以更细粒度地调整生成每个 position 的行为。 <br> - 在一些任务中（任务回复短、token 数少、模型不是极大规模/非 MoE）表现已经很好。                       | - 对 token‐级比率的方差敏感，特别当响应序列长或者输出多 token 时噪声大。 <br> - 在 MoE 模型中可能不稳定；可能需要额外机制（如 Routing Replay）。 <br> - 裁剪与奖励估计可能在 token 级别累积误差，导致训练不稳定或效率低下。 <br> - 随着任务规模／复杂度上升，其扩展性受限。                              |
| **GSPO** | - 序列级别优化更稳定，对 token 级的极端概率变化不敏感。 <br> - 在 MoE 模型中训练稳定性更好，不需要额外的路由回放等机制。 <br> - 效率更高，相同资源下性能更好且可持续提升；更适合大规模训练 /长序列任务。 <br> - 基础设施友好，因为简化了很多 token‐级别复杂性。 | - 序列级别优化可能在某些场景里丧失某些 token 级微调精细性的调整（比如每个 token 的微小修正）。 <br> 对奖励模型或任务设计中序列整体质量的依赖更强。如果整个响应的奖励设计不好，序列级优化可能弱于 token‐级别修正。 <br> 在某些任务中，对序列长度或内容差异更敏感，需要进行长度归一化等额外处理。 <br> 新算法相对较新，可能在某些场景 /特殊任务中未被广泛验证。 |


如果模型规模中等，任务输出较短（例如问答回复、对话、摘要等短序列），且没有使用 MoE 结构，那么 GRPO 通常已足够，并且可能在微调每个 token 行为上更灵活。
如果你正在训练非常大的语言模型 / MoE 模型 /输出序列很长 /任务依赖于完整输出质量（比如数学问题解决／编程／复杂推理输出），那么 GSPO 更加合适，因为它在定性、效率、 scaling（随着投入更多算力性能持续提升）方面表现更好。
在 MoE 模型或者当观察到训练中有不稳定的现象（如严重方差、策略崩塌、性能曲线震荡等），优先考虑 GSPO。

vLLM 不支持视觉/编码器层的 LoRA，因此在加载 LoRA 适配器时设置 finetune_vision_layers = False


本次更新还添加了 GSPO（ 组序列策略优化 ) 是阿里巴巴 Qwen 团队开发的 GRPO 的一个变体。他们注意到，尽管 GRPO 的显式优势不会随着每个 token 而扩展或变化，但它隐式地为每个 token 计算了重要性权重。


# Notes
Unsloth只支持微调LLM层的Lora参数，不支持Vision层的。

# 数据集： AI4Math/MathVista
结合图／图表／科学论文图／函数图／几何图形等视觉元素来解题
每个样本通常包含这些字段 /元数据 (metadata)：

question：问题文本。 
Hugging Face
+1

image：与问题相关的图像（图形／图表／几何图／科学图等）。 
遇见数据集
+1

decoded_image：图像的解码形式（方便模型读取／显示）。 
Hugging Face

choices：对于选择题（multiple choice）的问题有哪些选项。如果是开放式问题（free‐form），这个字段可能是 none 或空。 
遇见数据集
+1

unit：答案所带的单位（例如 “m^2” 等），如果无单位则标为 none。 
Hugging Face

precision：答案的小数精度（如果答案是浮点数的话）。 
Hugging Face

answer：正确答案。 
Hugging Face

question_type：问题类型，例如 “multi_choice” 或 “free_form”。 
Hugging Face

answer_type：答案类型，比如 “integer”，“float”，“text”，“list” 等。 
Hugging Face

metadata：包含很多额外信息，比如：

category：数学‐针对的 VQA ("math‐targeted‐vqa") 或一般的 VQA ("general‐vqa") etc. 
遇见数据集
+1

context：视觉上下文类型 (geometry diagram, scientific figure, synthetic scene, etc.) 
Hugging Face

grade：难度或学年级别（elementary, high school etc.） 
遇见数据集
+1

img_width / img_height：图像尺寸。 
遇见数据集
+1

language：问题语言，如英语、中文、波斯语 (Persian) 等。 
遇见数据集
+1

skills：这个题目测评哪些数学 /逻辑能力（例如算术推理、几何推理、科学推理、数字常识等） 
遇见数据集
+1

source：原始题目来自哪个数据集。 
遇见数据集
+1

split：属于 testmini 还是 test。 
Hugging Face

task：该题目的任务性质，例如 “geometry problem solving”, “visual question answering” 等。 
遇见数据集
+1

query：用于评估模型时的输入 prompt / 查询文本。可能与 question 略有不同，以兼顾评测格式。 
遇见数据集
+1

