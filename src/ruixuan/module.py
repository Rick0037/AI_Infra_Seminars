import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


# TODO：pytorch 模型默认使用kaiming 初始化， 输出实际上是水机的，等待后续加载别人的模型进行权重加载
model = SimpleModel(784, 256, 10)
# TODO： 注意不能使用forward， 要使用 model(x) 来进行operator()的实现
output = model(torch.randn(32, 784))
print(output.shape)


for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params}")  # 55


print(model.state_dict().keys())
# odict_keys(['linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias'])

linear1_weight = model.state_dict()["linear1.weight"]
print(linear1_weight.shape)
# print(linear1_weight)
