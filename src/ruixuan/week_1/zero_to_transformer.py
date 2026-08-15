import torch
import torch.nn as nn
import math


# 点积缩放 + mask 算子
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


# 输入/输出 （embedding 后、FFN 前）[B, L, H, d_k]
# 计算 Attention 时 [B, H, L, d_k]
# multi head attention 算子
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # 要求 d_model 可以被 n_heads 整除才行, 取余数为0
        assert d_model % n_heads == 0
        # 保存各种得参数
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads

        # 实际上都是方阵，前三个在后面会进行分头
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.output_c = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    # 有时候这里得q,k,v 是来自与不同得方面的
    def forward(self, q, k, v, mask=None):
        B, L, _ = q.size()  # 输入: [B, L, D]
        #  [B, L, Head, dk]
        Q = self.W_q(q).view(B, L, self.n_heads, self.dk)
        K = self.W_k(k).view(B, L, self.n_heads, self.dk)
        V = self.W_v(v).view(B, L, self.n_heads, self.dk)

        # 2. 调整维度准备并行计算
        # TODO [B, L, H, d_k] → Transpose → [B, H, L, d_k]
        # 现在H和L交换位置，方便在L维度上做注意力计算
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        output, distance = self.attn(Q, K, V, mask)
        # 得到了多头得结果， 之后进行合并
        # TODO  进行拼接
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        # TODO 拼接之后做一次线性的变换就可以了
        return self.output_c(output)


# 一般约定都是 ： d_ff ≈ 4 × d_model
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO linear 层默认是 XW + b， 默认是加bias的
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 实际上时d_model ->d_ff
            nn.ReLU(),
            nn.Linear(d_ff, d_model),  # d_ff -> d_model
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        # 可学习参数，初始gamma=1(不缩放)，beta=0(不平移)
        # nn.Parameter
        # 是一个 特殊的 Tensor 包装器 ，只有一个目的：
        # 告诉 PyTorch「 这是需要训练的参数，把它加到 model.parameters() 里，并记录梯度 」。
        # TODO parameter 代表可以被学习
        self.gamma = nn.Parameter(torch.ones(d_model))  # [D]
        self.beta = nn.Parameter(torch.zeros(d_model))  # [D]
        self.eps = eps  # 防止除0

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 求每一行的均值
        # TODO 指的是 用有偏估计 计算方差，分母是 N
        # 有偏 和无偏移的  unbiased=False 或者
        # unbiased=True 影响了被除的分母 到底时N 还是 N-1
        # var 代表方差 std 代表标准差
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * out + self.beta


"""
# layer norm的小实验
ln = LayerNorm(4)
# --- 验证 1: 它们确实在 parameters() 里
for name, p in ln.named_parameters():
    print(name, p.shape, p.data)
# gamma [4] tensor([1., 1., 1., 1.])
# beta  [4] tensor([0., 0., 0., 0.])
# --- 验证 2: 训练一轮后数值变了
x = torch.randn(2, 4)
optimizer = torch.optim.AdamW(ln.parameters(), lr=0.1)
loss = ln(x).sum()  # 随便构造一个 loss
loss.backward()  # 算梯度
optimizer.step()  # 更新参数
print(ln.gamma.data)  # [0.9000, 0.9000, 0.9000, 0.9000]  ← 从 1.0 变了
print(ln.beta.data)  # [0.1000, 0.1000, 0.1000, 0.1000]  ← 从 0.0 变了
"""


# PE(pose, 2i) = sin(pose / 10000 2i/dmoel)
# PE(pose, 2i+1) = cos(pose/ 10000 2i+1)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, D]

        # torch.arange(0, max_len) -> [max_len]
        # unsqueeze(1) -> [max_len, 1] 增加了一个维度
        # pos: [max_len, 1] - 位置索引列向量 [0,1,2...4999]^T
        pos = torch.arange(0, max_len).unsqueeze(1)

        # div_term: [D/2] - 频率衰减项，指数递减 TODO 一个一维的东西
        # torch.arange(0, d_model, 2) = [0, 2, 4, 6, 8]
        # 实际上是数学公式转过来的, 指数运算
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 广播计算: pos([max_len,1]) * div([D/2]) → [max_len, D/2]
        # 偶数维用sin，奇数维用cos
        # pe[:, 0::2] 针对每一行 以0为间隔
        # pe[:, 0::2] 针对每一行 以1为间隔
        pe[:, 0::2] = torch.sin(pos * div)  # [max_len, D/2]
        pe[:, 1::2] = torch.cos(pos * div)  # [max_len, D/2]

        # 注册为buffer:  pe.unsqueeze(0) 升级维度 [1, max_len, D]，第0维为batch维度
        # register_buffer 把一个张量 存进模块
        # 但它 不是参数 （不会被 optimizer 更新），而是 随模型一起保存/加载/迁移设备
        # TODO
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, L, D]
        # self.pe[:, :L]: [1, L, D]
        # 广播相加: [B, L, D] + [1, L, D] → [B, L, D]
        # pe在batch维广播，自动复制到所有样本
        # TODO x.size(1) 实际上就是 L
        # self.pe 上面的被unsqueeze 给增加了第一维，【1 max D】
        #  self.pe[:, : x.size(1)] 实际上对 self 的前两个维度进行了切片
        # [1, L, D]
        # BLD 和 1LD 就完全是复制了， 可以直接相加，因为PE 在不同batch中也是一样的
        return x + self.pe[:, : x.size(1)]


class Mask_Address:
    # Padding 掩码 ：标记哪些位置是真词，哪些是填充的 0
    # Encoder 自注意力
    def make_src_mask(self, src):
        # src: [B, L] 针对没一个句子 只有batch 以及length
        # 非零位置为True(有效词)，零位置为False(Padding)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        # 因为 Attention 的 score 形状是 [B, H, L, L] ，mask 要和它广播对齐：
        # TODO [B, 1, 1, L] 广播到 [B, H, L, L] ：每个 head、每个 query 位置，
        # 只留下了最后的一列，就是再query的时候不要看padding的哪些key

    # Padding + 因果 双掩码 ：既要屏蔽填充，又要屏蔽「未来词」
    # Decoder 自注意力
    def make_tgt_mask(self, tgt):
        # 非常常见的获取维度的方法
        B, L = tgt.size()
        # Padding掩码: [B, 1, 1, L]
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

        # 因果掩码（下三角）: [L, L]，上三角为False
        causal_mask = torch.tril(torch.ones(L, L)).bool()

        # 广播与运算:
        # pad_mask: [B, 1, 1, L] → 广播为 [B, 1, L, L]
        # causal_mask: [L, L] → 广播为 [B, 1, L, L]
        # 结果: 必须同时满足"非填充"且"不越界"
        return pad_mask & causal_mask  # [B, 1, L, L]

    # 广播运算 与padding 还有 上三角掩码同时计算出的结果
