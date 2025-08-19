# 智星云部署
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name areal ghcr.io/inclusionai/areal-runtime:v0.3.0.post2 sleep infinity
docker start areal
docker exec -it areal bash

# 检查安装结果
```
python examples/env/validate_installation.py
输出:
python examples/env/validate_installation.py
AReaL Installation Validation
==================================================

=== Testing Critical Dependencies ===
  - CUDA devices: 2
  - CUDA version: 12.8
✓ torch
  - transformers imported successfully
✓ transformers
  - Flash attention functions imported successfully
✓ flash_attn
✓ cugae
/sglang/python/sglang/srt/managers/session_controller.py:57: SyntaxWarning: invalid escape sequence '\-'
  prefix = " " * len(origin_prefix) + " \- " + child.req.rid
  - SGLang imported successfully
✓ sglang
✓ ray
✓ numpy
✓ scipy
✓ hydra
✓ omegaconf
✓ datasets
✓ pandas
✓ einops
✓ wandb
✓ pynvml
✓ aiohttp
✓ fastapi
✓ uvicorn
✓ sympy
✓ latex2sympy2

=== Testing Optional Dependencies ===
⚠ vllm (OPTIONAL): No module named 'vllm'
✓ grouped_gemm
✓ flashattn_hopper
✓ tensorboardX
✓ swanlab
✓ matplotlib
✓ seaborn
✓ numba
✓ nltk

=== Testing CUDA Extensions ===
✓ Basic CUDA operations working
✓ Flash attention CUDA operations working

==================================================
VALIDATION SUMMARY
==================================================
Total tests: 29
Successful: 28
Failed: 1

⚠️  WARNINGS (1):
  - vllm: No module named 'vllm'

✅ INSTALLATION VALIDATION PASSED
Note: Some optional dependencies failed but this won't affect
core functionality

```
# 安装areal
pip install --no-deps -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 测试
```
# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test flash attention
python -c "import flash_attn; print('Flash Attention: OK')"
python -c "import flashattn_hopper; print('Flash Attention Hopper: OK')"

# Test inference engines
python -c "import sglang; print('SGLang: OK')"
python -c "import vllm; print('vLLM: OK')"
```

# 训练
修改配置文件gsm8k_grpo.yaml的模型path为Qwen/Qwen2-0.5B-Instruct
[gsm8k_grpo.yaml](..%2F..%2FAReaL%2Fexamples%2Flite%2Fconfigs%2Fgsm8k_grpo.yaml)
actor:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: Qwen/Qwen2-0.5B-Instruct
开始训练
python3 -m areal.launcher.local examples/lite/gsm8k_grpo.py --config examples/lite/configs/gsm8k_grpo.yaml
