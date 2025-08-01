# verl/scripts/legacy_model_merger.py


是用于将 FSDP 或 Megatron 保存的模型 checkpoint 转换为 Hugging Face 的格式的工具脚本**，你找的就是它。如果你想将 `verl` 训练的模型（如 Qwen2.5-0.5B）保存为标准 Huggingface 格式模型，可以使用它。
---

## ✅ **核心功能**

支持两种后端：

* `fsdp`: 从 FSDP 保存的多个 checkpoint 中合并模型权重并转换为 Huggingface 格式。
* `megatron`: 从 Megatron 保存的分布式张量中恢复权重，按 transformer 层和张量并行（TP）方式合并。

支持两种操作：

* `merge`: 合并并转换保存为 Huggingface 格式模型。
* `test`: 用来测试合并出来的权重与参考 Huggingface 模型的权重是否一致。

---

## ✅ **怎么使用这个脚本**

假设你训练的是一个基于 FSDP 的模型，最终保存路径如下：

```
checkpoints/my_qwen_model/global_step_1/actor/
```

该目录下有类似如下的文件：

```
model_world_size_2_rank_0.pt
model_world_size_2_rank_1.pt
...
```

你可以用下面命令来将其合并为 Huggingface 格式：

```bash
python scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir checkpoints/my_qwen_model/global_step_1/actor \
  --target_dir /path/to/output_hf_model
```

其中：

* `--backend fsdp`：说明是用 FSDP 保存的。
* `--local_dir`：FSDP 模型保存路径。
* `--target_dir`：你希望输出 Huggingface 模型的路径。

输出的 Huggingface 模型将包括：

* `pytorch_model.bin` 或 `model.safetensors`
* `config.json`
* `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json`
* 可选的 `generation_config.json`

---

## ✅ **如果你用的是 Megatron 保存的 checkpoint**

示例命令如下：

```bash
python scripts/legacy_model_merger.py merge \
  --backend megatron \
  --tie-word-embedding \
  --local_dir checkpoints/my_qwen_model/global_step_1/actor \
  --target_dir /path/to/output_hf_model
```

* `--tie-word-embedding` 是针对 Megatron 模型的一个特殊处理选项，表示共享词嵌入参数（word embedding 和 lm\_head 权重共享）。

---

## ✅ **测试模式（test）**

你可以验证转换出来的 Huggingface 模型是否与原始 Huggingface 模型一致（例如 `FSDPCheckpointManager` 保存出来的 HF 模型）：

```bash
python scripts/legacy_model_merger.py test \
  --backend fsdp \
  --local_dir checkpoints/my_qwen_model/global_step_1/actor \
  --test_hf_dir /path/to/reference_hf_model
```

---

## ✅ 使用流程总结

| 任务                       | 命令模板                                                                                                           |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| 合并 FSDP 模型并转换为 HF 格式     | `python legacy_model_merger.py merge --backend fsdp --local_dir ... --target_dir ...`                          |
| 合并 Megatron 模型并转换为 HF 格式 | `python legacy_model_merger.py merge --backend megatron --tie-word-embedding --local_dir ... --target_dir ...` |
| 验证转换正确性                  | `python legacy_model_merger.py test --backend fsdp --local_dir ... --test_hf_dir ...`                          |

---

## 如果你想用这个脚本：

你需要满足以下依赖环境：

* 安装 `transformers`
* 安装 `torch`, `safetensors`, `accelerate`, `tqdm`
* `verl` 自带的一些工具（`verl.utils` 下的 tokenizer 和 processor），你已经在使用 Verl 的话应该都有。