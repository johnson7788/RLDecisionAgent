# 猜数字（决策Agent）
多轮对话

## Step1： 生成数据
python generate_data.py

## Step3: SFT训练模型， 查看ray进度: ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -NTf -L 8265:127.0.0.1:8265 -p 1008 johnson@lgr
bash train_sft.bash
