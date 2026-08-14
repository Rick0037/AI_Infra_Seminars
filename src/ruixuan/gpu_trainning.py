"""
import torch
import torch.nn as nn
from torch.optim import optimizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Linear(100, 10).to(device)
x = torch.randn(32, 100).to(device)
y = model(x)

# -1 代表 cpu， 0 代表gpu
print(y.get_device())


device = torch.device("cuda")
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 10)
).to(device)
optimizer = torch.optim.AdamaW(model.parameters(), lr=1e-3)

# dtype=torch.bfloat16   # ← 推荐:Ampere+ GPU(A100/5090/4090)，不需要 GradScaler
# dtype=torch.float16    # ← 旧 GPU(V100/2080Ti)，必须配 GradScaler
# dtype=torch.float16    # CPU 上只支持 float16，不支持 bfloat16 的 autocast

# BF16 混合精度（推荐，Ampere+ GPU，不需要 GradScaler）
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    # 计算模型以及的时候继续宁autocase，但是在后面更新权重的时候不要加
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# FP16 混合精度（需要 GradScaler 防止梯度下溢）
scaler = torch.amp.GradScaler("cuda")


for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = nn.functional.cross_entropy(model(images), labels)
    # loss 放大 * 乘以一个很大的数值
    scaler.scale(loss).backward()
    # 成比例的再缩小回去，更新权重
    scaler.step(optimizer)
    # 动态的更新放大的比例
    scaler.update()
    optimizer.zero_grad()
"""

import torch
import torch.nn as nn

assert torch.cuda.is_available()
device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()
data = torch.randn(256, 1024, device=device)
labels = torch.randint(0, 10, (256,), device=device)
# 生成256 个 从 0 到 10的随机数字


# 模型也要搬运到gpu上面
def make_model():
    return nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    ).to(device)


# fp32 训练
model_fp32 = make_model()
opt_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3)
torch.cuda.reset_peak_memory_stats()
loss = criterion(model_fp32(data), labels)
loss.backward()
opt_fp32.step()
opt_fp32.zero_grad()
fp32_peak = torch.cuda.max_memory_allocated() / 1024**2


# bf16 混合精度训练
model_bf16, opt_bf16 = make_model(), None
opt_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=1e-3)
torch.cuda.reset_peak_memory_stats()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = criterion(model_bf16(data), labels)
loss.backward()
opt_bf16.step()
opt_bf16.zero_grad()
bf16_peak = torch.cuda.max_memory_allocated() / 1024**2


print(
    f"FP32: {fp32_peak:.1f} MB | BF16: {bf16_peak:.1f} MB | 节省: {(1-bf16_peak/fp32_peak)*100:.1f}%"
)
