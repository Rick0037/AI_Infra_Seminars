import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    # 基本上都是有self的
    # 输入的mask 是我们自己填好的矩阵，上三角单不包括对角线
    def forward(self, Q, K, V, mask=None):
        # [batch, heads, seq, d_k]   ← PyTorch 默认,计算高效
        # [batch, seq, heads, d_k]   ← 直觉上好理解,但计算前要转
        # -1 代表的都是最后一个维度,
        d_k = Q.size(-1)
        d_v = V.size(-1)
        # TODO 一般最后遗留下来的都是 [seq d_k]
        # [batch, heads, seq, d_k] * [batch, heads, d_k, seq] -> [seq, seq]
        distance = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # 加 mask(在 softmax 之前!)
        # -1e9 是否也可以
        if mask is not None:
            distance = distance.masked_fill(mask == 0, -1e9)
        # 针对每一行进行softmax
        # 一个代表着有几行，另外一个代表着每行有几个元素， 实际上 -1 代表着某一行多少个元素得维度，
        # 实际上从哪个 d_k  = Q.size(-1) 也能看出来，实际上是一个东西
        distance = torch.softmax(distance, dim=-1)
        # 可以只返回第一个啊
        return torch.matmul(distance, V), distance
