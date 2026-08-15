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


"""
# TODO：pytorch 模型默认使用kaiming 初始化， 输出实际上是水机的，等待后续加载别人的模型进行权重加载
model = SimpleModel(784, 256, 10)
# TODO： 注意不能使用forward， 要使用 model(x) 来进行operator()的实现
output = model(torch.randn(32, 784))
print(output.shape)


for name, param in model.named_parameters():
    print(f"{name} : {param.shape}")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params}")  # 55


print(model.state_dict().keys())
# odict_keys(['linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias'])

linear1_weight = model.state_dict()["linear1.weight"]
print(linear1_weight.shape)
# print(linear1_weight)

print(model.state_dict())
print(model.state_dict().keys())

# TODO: @ruixuan 512 input
# TODO: @ruixuan 256 output
linear = nn.Linear(512, 256)

# num_embeddings 代表的单词表的大小， 不是position embedding
# embedding_dim 代表的是单个单词的维度， 每个单词都是768 维度的
embedding = nn.Embedding(num_embeddings=50000, embedding_dim=768)

# 实际上这里都是通过  tesnor 对应的token id 来进行索引的
input_token = embedding(torch.tensor([0, 21]))  # [2， 768]
print(input_token.shape)

# 实际上做模型像是在搭积木，
# 单层的layer norm
layer_norm = nn.LayerNorm(768)

# p=0.1 表示 训练时每个神经元有 10% 的概率被丢弃（置为 0） ，剩下的值会乘以 1/(1-p) 放大。
dropout = nn.Dropout(p=0.1)

# 通过 model.train(), model.eval() 来进行训练模式与推理模式的区分
"""


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


model2 = TwoLayerMLP(784, 256, 10)
print(f"参数量: {sum(p.numel() for p in model2.parameters()):,}")  # 203,530
print(model2(torch.randn(64, 784)).shape)  # torch.Size([64, 10])
