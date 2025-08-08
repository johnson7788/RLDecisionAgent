# 智星云部署

测试
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