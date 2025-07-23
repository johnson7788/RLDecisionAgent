# 猜数字（决策Agent）

## Step1： 生成数据
python generate_data.py

## Step2: 预处理生成parquet格式
python preprocess_data.py

## Step3: 训练模型， 查看ray进度: ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -NTf -L 8265:127.0.0.1:8265 -p 1008 johnson@lgr
bash train.bash
