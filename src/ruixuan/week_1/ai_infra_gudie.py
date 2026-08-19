from math import cos
from turtle import forward

import numpy as np
import torch
import torch.nn as nn

from ruixuan.week_1.transformer_main import d_k


# each line have their padding position
def get_len_mask(b, max_len, feat_lens, device) -> torch.Tensor:
    #
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        # attn_mask[i][:, : feat_lens[i]] = 0
        attn_mask[i, :, : feat_lens[i]] = 0
    return attn_mask.to(torch.bool)


# 上三角的向后屏蔽的掩码mask
def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    """生成 Decoder 自注意力的因果掩码（上三角矩阵）。
    Returns: shape (b, max_len, max_len)，True 表示未来位置（需屏蔽）
    """
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(
        torch.bool
    )


# enc dec 同时存在mask的情况，这种情况和decoder一样需要屏蔽无意义的padding
def get_enc_dec_mask(b, max_enc_len, feats_len, max_dnc_len, device) -> torch.Tensor:
    """_summary_
    Args:
        b (_type_): batch size
        max_enc_len (_type_): encoder max length
        feats_len (_type_): encoder length
        max_dnc_len (_type_): decoder max length
        labels_lens (_type_): decoder length
        device (_type_): device is cuda or cpu

    Returns:
        torch.Tensor: mask output tensor
    """
    #!TODO: 实际上dnc 这里是列，所以需要在前面
    attn_mask = torch.ones((b, max_dnc_len, max_enc_len), device=device)
    for i in range(b):
        # 针对每一个batch句子，将一列feature_len[i] 之后的列都进行屏蔽，要求query不去看他们
        attn_mask[i, :, feats_len[i]] = 0
    return attn_mask.to(torch.bool)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.0):
        """
        Args:
            d_k: 每个注意力头的 Key/Query 维度
            d_v: 每个注意力头的 Value 维度
            d_model: 输入/输出的总维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads

        self.WQ = nn.Linear(d_model, d_k * num_heads)
        self.WK = nn.Linear(d_model, d_k * num_heads)
        self.WV = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        # 权重初始化（参考 He initialization 思路）
        nn.init.normal_(self.WQ.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.WK.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.WV.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, mask):
        """
        Args:
            Q: (batch, q_len, d_model)
            K: (batch, k_len, d_model)
            V: (batch, v_len, d_model)  注意 k_len == v_len
            attn_mask: (batch, q_len, k_len) 或 None
        Returns:
            output: (batch, q_len, d_model)
        """
        batch_size = Q.size(0)
        # [b, len, heads, dk]
        Q = self.WQ(Q).view(batch_size, -1, self.num_heads, self.d_k)
        # [b, heads, len, dk]
        Q = Q.transpose(1, 2)
        # [b, len, heads, dk]
        K = self.WK(K).view(batch_size, -1, self.num_heads, self.d_k)
        # [b, heads, len, dk]
        K = K.transpose(1, 2)
        # [b, len, heads, dv]
        V = self.WV(V).view(batch_size, -1, self.num_heads, self.d_v)
        # [b, heads, len, dv]
        V = V.transpose(1, 2)

        # Step 2: 广播 Mask 到 head 维度
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            # 实际 不一定需要 repeat， repeat 代表的是自动进行广播
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1).bool()

        score = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(d_k)

        # 实际上是一个【True，False】的一个矩阵
        if attn_mask is not None:
            score.masked_fill(attn_mask, -1e4)
            # 按照行进行softmax
        attn = torch.softmax(score, dim=-1)
        # attn = torch.dropout(attns)

        # score [n,n] * [n, 这里不是简单的d_v 而是concat起来的 dv 都在一起了] ->
        # v -> [b, heads, len, dv]
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).view(batch_size, -1, self.d_v * self.num_heads)

        return self.W_out(output)


def Pe(seq_len: int, d_model: int) -> torch.Tensor:
    """_summary_

    Args:
        seq_len (int): _description_
        d_model (int): _description_

    Returns:
        torch.Tensor: 一次性返回整个sequence的整个postion embedding
    """
    res = torch.zeros(seq_len, d_model)
    pose = torch.arange(0, seq_len)
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        # 可以实际使用 exp 加 ln 的计算把表达式进行简化
        res[:, i] = f(pose / np.power(1e4, 2 * (i // 2) / d_model))
    return res.float()


class FFN(nn.Module):
    def __init__(self, d_model, d_ff) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # self.linear_1 = nn.Linear(d_model, d_ff)
        # self.linear_2 = nn.Linear(d_ff, d_model)
        # self.relu = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, mask) -> None:
        super().__init__()
        self.d_k = d_model // n_heads

        self.attn = MultiHeadAttention(d_k, d_k, d_model, n_heads)
        self.norm_1 = nn.LayerNorm(dim)
        self.ffn = FFN(d_model, d_ff)
        self.norm_2 = nn.LayerNorm(dim)

    def forward(self, enc_in, mask):
        score = self.attn(enc_in, enc_in, enc_in, mask)
        score = self.norm_1(score + enc_in)

        output = self.ffn(score)
        return self.norm_2(output + score)


class Encoder(nn.Module):
    def __init__(self, n_layers, enc_dim, num_head, d_ff, max_res_len) -> None:
        super().__init__()
        self.tgt_len = max_res_len
        self.pos_emb = nn.Embedding.from_pretrained(
            Pe(max_res_len, enc_dim), freeze=True
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim, num_head, dff)] for _ in range(n_layers)
        )

    def forward(self, X, mask):
        # X: (batch, seq_len, d_model)
        seq_len = X.size(1)
        output = X + self.pos_emb(torch.arange(seq_len, device=X.device))
        for _ in range(self.layers):
            out = layer(out, mask)
        return out


class DecoderBlock(nn.Module):
    # 主要都是参数定义，比如d_model, d_ff, heads
    def __init__(self, d_model, d_ff, h_heads) -> None:
        super().__init__()
        d_k = d_model // h_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.poswise_ffn = FFN(d_model, d_ff)
        self.dec_attn = MultiHeadAttention(
            d_k, d_k, d_model, h_heads
        )  # Masked Self-Attn
        self.enc_dec_attn = MultiHeadAttention(d_k, d_k, d_model, h_heads)  # Cross-Attn

    # 主要是实际的输入，比如enc_out, dec_int
    def forward(self, enc_out, dec_in, mask_1, mask_2):
        # mask_2 是掩码mask
        atten1 = self.dec_attn(dec_in, dec_in, dec_in, mask_2)
        layer_1_out = self.norm1(dec_in + atten1)
        atten2 = self.dec_attn(layer_1_out, enc_out, enc_out, mask_1)
        layer_2_out = self.norm2(layer_1_out + atten2)
        atten3 = self.poswise_ffn(layer_2_out)
        return self.norm3(layer_2_out + atten3)


class Decoder(nn.Module):
    def __init__(
        self,
        dropout_emb,
        num_layers,
        dec_dim,
        num_heads,
        dff,
        tgt_len,
        tgt_vocab_size,
    ):
        super(Decoder, self).__init__()
        # Word Embedding：将 token ID 映射为 d_model 维向量
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        # 固定正弦位置编码
        self.pos_emb = nn.Embedding.from_pretrained(Pe(tgt_len, dec_dim), freeze=True)
        self.layers = nn.ModuleList(
            [DecoderBlock(dec_dim, num_heads, dff) for _ in range(num_layers)]
        )

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask):
        # labels: (batch, dec_len) token ID 序列
        # TODO by ruixuan decoder 的最终output 实际上就是 dec_out， 和输入一个维度的tensor
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)
        for layer in self.layers:
            dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        return dec_out


class Transformer(nn.Module):
    # frontend 前端实际上是一个处理embedding的地方
    def __init__(self, frontend, encoder, decoder, output_dim, vec_dim) -> None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.output = nn.Linear(output_dim, vec_dim)

    def forward(self, x, x_lens, labels, label_lens):
        # b 一共有多少个批次
        # enc_len 一个批次中最多有多少个但粗
        # 实际批次都是 x_lens[i] 个单词

        # x [b, enc_len, d_model]
        # x_lens [b] each batch line sequence length
        out = self.frontend(x)  # [B, enc_len', d_model]
        b = out.size(0)
        max_enc_seq_lens = out.size(1)

        enc_mask = get_len_mask(b, max_enc_seq_lens, x_lens)  # [B, enc_len', enc_len']
        enc_out = self.encoder(out, enc_mask)  # [B, enc_len', d_model]

        # Decoder（对应第 4 节 Decoder 结构）
        max_label_len = labels.size(1)
        # 上三角矩阵的mask
        dec_mask = get_subsequent_mask(b, max_label_len)
        # TODO：按照列的masked， 实际上两个mask应该与的，这里在维度不统一的时候
        # mask 矩阵应该是【dec_dim, enc_output_dim】 想办法把 enc_output_dim中的列mask掉
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)

        # LM Head
        return self.linear(dec_out)
