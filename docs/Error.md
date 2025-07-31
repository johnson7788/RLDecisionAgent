# 错误收集

## 1. 显卡
Tesla V100-SXM2-32GB 不支持
flash_attn_gpu.varlen_fwd(
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: FlashAttention only supports Ampere GPUs or newer.

## 2. 代理
env_file配置代理，然后pycharm中的Paths to env加载env文件，即可使用代理
```
ALL_PROXY=http://127.0.0.1:7890
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```