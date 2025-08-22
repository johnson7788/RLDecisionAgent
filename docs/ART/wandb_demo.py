#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/23 07:00
# @File  : wandb_demo.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
# 先配置环境变量和登录
# export WANDB_BASE_URL="http://192.168.100.8:3005"
# wandb login --host=http://192.168.100.8:3005
# 输入自己的key
# python wandb_demo.py
# wandb: (1) Create a W&B account
# wandb: (2) Use an existing W&B account
# wandb: (3) Don't visualize my results
# wandb: Enter your choice: 2
# wandb: You chose 'Use an existing W&B account'
# wandb: You can find your API key in your browser here: http://192.168.100.8:3005/authorize?ref=models
# wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:

"""
wandb_demo.py
一个“尽量全面”的 Weights & Biases 使用示例（PyTorch + MNIST）：
- wandb.init / config / log 基本用法
- 自动系统指标；模型梯度/参数直方图与计算图（run.watch）
- 训练/验证/测试指标
- 记录样例图片 (wandb.Image)
- 记录预测表 (wandb.Table)
- 混淆矩阵 (wandb.plot.confusion_matrix)
- 模型作为 Artifact 版本化与别名（best / epoch-#）
兼容本地 W&B Server：尊重 WANDB_BASE_URL / WANDB_MODE 等环境变量

* `wandb.init` 行为（后台进程、在线/离线/禁用模式、脚本结束或 `finish()` 收尾）([Weights & Biases Documentation][2])
* `wandb.log` 可记录标量/媒体/表格；系统指标自动记录（CPU/GPU、stdout/stderr）([Weights & Biases Documentation][1])
* 图片与其他媒体类型的记录方式（`wandb.Image` 等）([Weights & Biases Documentation][4])
* `wandb.Table` 的创建、增量添加与记录（测试预测表）([Weights & Biases Documentation][5])
* `wandb.plot.confusion_matrix` 自定义图表 API（直接传 `y_true`/`preds`）([Weights & Biases Documentation][6])
* 模型作为 **Artifact** 版本化与通过 `run.log_artifact()` 保存（别名 `best`/`epoch-#`）([Weights & Biases Documentation][7])
* PyTorch 集成以及 `run.watch(model, log="all", log_freq=...)` 记录梯度/参数/计算图的示例与说明([Weights & Biases Documentation][8])

"""
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb


# ----------------------------
# 配置与随机种子
# ----------------------------
@dataclass
class DefaultCfg:
    project: str = "wandb-demo"
    entity: str | None = None        # 团队/组织名，可留空
    data_dir: str = "./data"
    batch_size: int = 128
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    log_freq: int = 100              # watch 的梯度/参数记录频率
    log_every: int = 50              # 训练中逐步日志频率（步）
    seed: int = 42
    num_workers: int = 2
    limit_samples: int | None = None # 为了快速演示，可限制每个集的样本数


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# 简单 CNN 模型
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ----------------------------
# 数据加载
# ----------------------------
def build_loaders(data_dir: str, batch_size: int, num_workers: int = 2,
                  limit_samples: int | None = None):
    tfm = transforms.Compose([transforms.ToTensor()])
    trainval = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

    if limit_samples:
        trainval = torch.utils.data.Subset(trainval, list(range(min(limit_samples, len(trainval)))))
        test = torch.utils.data.Subset(test, list(range(min(limit_samples // 5 if limit_samples else len(test), len(test)))))

    train_size = int(0.9 * len(trainval))
    val_size = len(trainval) - train_size
    train_set, val_set = random_split(trainval, [train_size, val_size], generator=torch.Generator().manual_seed(123))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ----------------------------
# 训练与评估
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total, correct = 0, 0
    losses = []
    all_true, all_pred = [], []
    sample_triplets: List[Tuple[torch.Tensor, int, int]] = []

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()

        # 仅采样前几个样例作图片展示
        if batch_idx == 0:
            for i in range(min(16, x.size(0))):
                sample_triplets.append((x[i].detach().cpu(), int(y[i].cpu()), int(pred[i].cpu())))

        all_true.extend(y.detach().cpu().tolist())
        all_pred.extend(pred.detach().cpu().tolist())

    return float(np.mean(losses)), correct / total, all_true, all_pred, sample_triplets


def train_one_run(cfg: DefaultCfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    # 1) 初始化 run（支持 online/offline/disabled 与本地 Server）
    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        config=vars(cfg),
        # mode 也可通过环境变量 WANDB_MODE 控制，此处尊重环境变量，不强制设置
        # mode=os.getenv("WANDB_MODE", None),
        save_code=True,  # 保存当前代码快照，便于复现（需账户设置允许）
        job_type="train",
        reinit=True,
    )

    # 2) 数据与模型
    train_loader, val_loader, test_loader = build_loaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.limit_samples
    )
    model = SimpleCNN(dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 3) 记录梯度/参数直方图与计算图
    #    注意：需要发生 forward/backward 并调用 wandb.log() 后，这些直方图等信息才会出现
    run.watch(model, log="all", log_freq=cfg.log_freq, log_graph=True)  # 梯度、参数、计算图
    global_step = 0
    best_val_acc = 0.0

    # 4) 训练循环
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                epoch_loss += loss.item() * y.size(0)
                epoch_correct += (pred == y).sum().item()
                epoch_total += y.size(0)

            if global_step % cfg.log_every == 0:
                wandb.log({"train/loss": loss.item()}, step=global_step)
            global_step += 1

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        # 5) 验证 + 混淆矩阵 + 预测样例图
        val_loss, val_acc, y_true, y_pred, samples = evaluate(model, val_loader, device)

        # 图片：用 wandb.Image 记录若干验证样例
        imgs = [wandb.Image(img, caption=f"truth={t}, pred={p}") for (img, t, p) in samples]
        # 混淆矩阵（支持传 probs 或 preds，这里用 preds/y_true）
        cm = wandb.plot.confusion_matrix(
            y_true=y_true, preds=y_pred, class_names=[str(i) for i in range(10)], title="Val Confusion Matrix"
        )

        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/samples": imgs,
            "val/confusion_matrix": cm
        }, step=global_step)

        # 6) 保存最佳模型并作为 Artifact 记录与打别名
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/best_model.pt"
            torch.save(model.state_dict(), ckpt_path)

            at = wandb.Artifact(name="mnist-cnn", type="model", metadata={"val_acc": best_val_acc, "epoch": epoch})
            at.add_file(ckpt_path)
            run.log_artifact(at, aliases=["best", f"epoch-{epoch}"])
            wandb.run.summary["best_val_acc"] = best_val_acc

    # 7) 在测试集上构建预测表（Tables）
    model.eval()
    test_table = wandb.Table(columns=["id", "image", "truth", "pred"] + [f"score_{i}" for i in range(10)])
    with torch.no_grad():
        idx = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for i in range(x.size(0)):
                row = [idx, wandb.Image(x[i].cpu()), int(y[i].cpu()), int(preds[i].cpu())] + \
                      [float(probs[i, j].cpu()) for j in range(10)]
                test_table.add_data(*row)
                idx += 1

    wandb.log({"test/predictions": test_table})
    run.finish()


if __name__ == "__main__":
    # 默认配置写在 DefaultCfg；Sweeps 覆盖时会自动注入到 wandb.config
    cfg = DefaultCfg()
    train_one_run(cfg)
