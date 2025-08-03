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


# converter_hf_to_mcore.py
converter_hf_to_mcore.py 是一个用于将 HuggingFace Transformers 格式的模型权重转换为 Megatron Core (mcore) 格式权重的脚本，支持分布式和单机转换，主要用于大模型（如 DeepseekV3、Qwen2/3 MoE、Qwen2.5 VL 等）的权重格式迁移。

主要功能与流程如下：

参数解析
使用 argparse 解析命令行参数，包括 HuggingFace 模型路径、输出路径、是否用 CPU 初始化、是否测试转换、是否信任远程代码等。

分布式环境初始化
自动检测并初始化 torch.distributed 分布式环境，设置 pipeline 并行等参数。

模型配置与分片
读取 HuggingFace 配置，判断是否支持分布式转换，并根据 world_size 计算 pipeline 分片。

Megatron Core 配置与模型初始化
通过 hf_to_mcore_config 生成 mcore 配置，调用 get_model 初始化 Megatron Core 模型。

HuggingFace 模型加载
根据模型类型（如 Qwen2.5 VL、Qwen2 MoE、DeepseekV3 等）加载对应的 HuggingFace 模型。

权重转换
针对不同模型结构，调用不同的转换函数（如 convert_checkpoint_from_transformers_to_megatron、convert_checkpoint_from_transformers_to_megatron_qwen2_5_vl、convert_checkpoint_from_transformers_to_megatron_dpskv3），将 HuggingFace 权重逐层拷贝到 mcore 模型。

权重保存
使用 Megatron Core 的分布式 checkpoint 工具保存转换后的权重到指定输出路径。

转换测试（可选）
若指定 --test，会加载保存的权重并与当前模型权重做一致性校验。

辅助函数说明：

safe_copy：安全地将一个 tensor 的数据拷贝到另一个 tensor，并做 shape/dtype 检查。
support_distributed_convert：判断当前模型是否支持分布式转换。
noop_context：空上下文管理器，用于兼容不同初始化方式。
适用场景：

适用于需要将 HuggingFace 格式大模型权重迁移到 Megatron Core 体系下，支持多种主流大模型结构，且支持分布式高效转换。
典型用法：

python converter_hf_to_mcore.py --hf_model_path <hf_model_dir> --output_path <mcore_output_dir>
或分布式：

torchrun --nproc_per_node 1 --nnodes 4 --node_rank <RANK> converter_hf_to_mcore.py --hf_model_path <hf_model_dir> --output_path <mcore_output_dir>
总结：
本脚本是 HuggingFace 到 Megatron Core 权重格式转换的自动化工具，支持多模型、多并行场景，便于大模型训练和推理体系的切换。