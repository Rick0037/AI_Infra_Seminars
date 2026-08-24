import numpy as np
import torch

""" int8 量化
"""


# def quant_per_tensor_absmax(x, n_bit=8):
#     scales = x.abs().max()
#     q_max = 2 ** (n_bit - 1) - 1
#     scales.clamp_(min=1e-5).div_(q_max)
#     q_x = x / scales
#     q_x = q_x.clamp_(-q_max, q_max).round_()
#     return q_x, scales


# def dequant(q_x, scales):
#     return q_x * scales


# X = torch.rand(2, 3, dtype=torch.float32)
# W = torch.rand(3, 4, dtype=torch.float32)

# # print(X)
# # print(X.shape)

# # print(W)
# # print(W.shape)

# q_x, x_scale = quant_per_tensor_absmax(X)
# q_w, w_scale = quant_per_tensor_absmax(W)
# q_y = torch.matmul(q_x, q_w)

# Y_head = dequant(q_y, x_scale * w_scale)

# Y = torch.matmul(X, W)
# print(Y)

# print(Y_head)
