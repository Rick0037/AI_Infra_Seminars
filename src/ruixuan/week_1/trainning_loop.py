'''
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
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# torchvision 专门做计算机视觉相关的库
from torchvision import datasets, transforms


batch_size, lr, num_epochs = 128, 1e-3, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 数据
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_set = datasets.MNIST("./data", train=True, download=False, transform=transform)
test_set = datasets.MNIST("./data", train=False, transform=transform)
# num_workers 过小会导致 GPU 等数据（data loading bottleneck）
# pin_memory 代表的是dma， 避免cpu进行copy
# shuffle 样本的数据被随机打乱，以这种方式进行训练
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)

test_loader = DataLoader(
    test_set, batch_size=batch_size, num_workers=2, pin_memory=True
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO nn.Sequential() 串行多个算子，避免首先所有的算子
        # TODO nn.Flatten() 除了batch 都进行亚平  [batch, 1, 28, 28] -> [batch, 784]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# MLP 搬运到了cuda上面
model = MLP().to(device)
# 计算loss的方法
criterion = nn.CrossEntropyLoss()
# adamw 实际优化器选择的哪个，TODO 将梯度同步权重，学习速率
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# 调度器， 根据num epochs 来进行多少批次的调度的优化
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# 训练
best_acc = 0.0  # 跟踪历史最优准确率,只保存最优权重
for epoch in range(num_epochs):
    model.train()
    # 打开model 中的dropout
    total_loss = 0
    for images, labels in train_loader:
        # 所有的计算都需要进行搬运, 搬运到cuda上去
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss item is {loss.item()}")
        total_loss += loss.item()
    scheduler.step()  # 再epoch循环的时候更新学习率

    # 验证
    # 关闭 dropout
    model.eval()
    correct = 0
    with torch.no_grad():  # 关闭计算图，简化后续直接掉model进行计算
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # TODO 手动比较法，比较两边的true or false 都是什么样的
            correct += (model(images).argmax(1) == labels).sum().item()
    acc = correct / len(test_set) * 100
    print(
        f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} "
        f"| Acc: {acc:.2f}%"
    )
    # # 保存 checkpoint
    # # TODO 为什么再训练中进行权重的保存，因为不知道是否发生了过拟合，所以ke
    # torch.save(
    #     {
    #         "epoch": epoch,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #     },
    #     f"ckpt_ep{epoch+1}.pt",
    # )
    # 只在验证准确率提升时保存最优权重(防止过拟合后的差权重覆盖好权重)
    if acc > best_acc:
        best_acc = acc
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acc": acc,
            },
            "best_model.pt",
        )
        print(f"  ↑ 新最优,已保存 best_model.pt (Acc={acc:.2f}%)")


print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"峰值:   {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(torch.cuda.memory_summary(abbreviated=True))

# 分析某段代码的显存
# 代码显示存储清0
torch.cuda.reset_peak_memory_stats()
# ... 运行目标代码 ...
print(f"峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# ==================== 训练结束后:导出最优模型为 ONNX ====================
"""
# 加载最优权重
best_ckpt = torch.load("best_model.pt", map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()

# 造一个 dummy 输入,告诉 ONNX 模型的输入形状
dummy_input = torch.randn(1, 1, 28, 28).to(device)  # [batch, channel, H, W]

# 导出 ONNX
torch.onnx.export(
    model,  # 要导出的模型
    dummy_input,  # 示例输入(用于追踪计算图)
    "best_model.onnx",  # 输出文件名
    input_names=["input"],  # 输入节点名
    output_names=["output"],  # 输出节点名
    dynamic_axes={  # 支持动态 batch_size
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=17,  # ONNX 算子版本
)
print(f"\n最优模型已导出为 best_model.onnx (最优 Acc={best_acc:.2f}%)")
"""
