import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SimpleDataSet(Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


# TODO 0 随机数字的下界
# TODO 10 随机数字的上界
# TODO 1000 生成1000 个数字
random_int = torch.randint(0, 10, (1000,))
print(random_int)
print(random_int.shape)
# data [1000, 784]
# label = [1000]
dataset = SimpleDataSet(torch.randn(1000, 784), random_int)

loader = DataLoader(
    dataset,  # Dataset: 传给 loader 的数据集对象
    batch_size=64,  # 每个批次包含 64 个样本
    shuffle=True,  # 每个 epoch 开始时打乱样本顺序
    num_workers=4,  # 用 4 个子进程并行加载数据
    pin_memory=True,  # 把数据放在锁页内存,加速 CPU→GPU 拷贝，走异步的DMA 直接内存访问
    drop_last=True,  # 丢弃最后不足 batch_size 的不完整批次
)


# --------------------------------
"""
device = "cuda"
for inputs, labels in loader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()  # ⑤ 清空梯度(放开头更常见)
    outputs = model(inputs)  # ① 前向传播
    loss = criterion(outputs, labels)  # ② 算损失
    loss.backward()  # ③ 反向传播算梯度
    optimizer.step()  # ④ 用梯度更新参数

    # 可选:打印监控
    if step % 100 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")
"""

model = nn.Linear(512, 10)

# TODO:
# lr = learning rate
# adamw 是自适应的权重衰减
# sgd 是一个完全单一步长的东西
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# TODO: 阶梯形状的东西，比如说固定多少epoch 就loss直接下降一倍
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# TODO: 以余弦函数作为一个技术，从多少进行cos的变换，loss下降大概率是一个平滑的曲线
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ==================== 断点续训:L70之后只改这里 ====================
import os

criterion = nn.CrossEntropyLoss()
CHECKPOINT_PATH = "checkpoint.pt"
start_epoch = 0
best_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(
        f"恢复 checkpoint:从 epoch {start_epoch} 继续,上次 loss={checkpoint['avg_loss']:.4f}"
    )
else:
    print("未找到 checkpoint,从头开始训练")

for epoch in range(start_epoch, 100):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_loss = running_loss / max(num_batches, 1)
    print(
        f"epoch {epoch:3d}  avg_loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}"
    )

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "avg_loss": avg_loss,
                "best_loss": best_loss,
            },
            CHECKPOINT_PATH,
        )


# 保存（模型 + 优化器 + 训练进度，断点恢复需要全部保存）

torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    },
    "checkpoint.pt",
)

# 加载

ckpt = torch.load("checkpoint.pt", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
# ckpt = torch.load("checkpoint.pt", weights_only=False)
# model.load_state_dict(ckpt["model_state_dict"])
# optimizer.load_state_dict(ckpt["optimizer_state_dict"])
