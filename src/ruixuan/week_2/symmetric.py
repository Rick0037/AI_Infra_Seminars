import numpy as np
import torch
from torch import nn as nn

""" 
# eager graph + dynamic
m = nn.Sequential(nn.Conv2d(2, 64, (8,)), nn.ReLU(), nn.Linear(16, 10), nn.LSTM(10, 10))

m.eval()

from torch.quantization import quantize_dynamic

model_quantization = quantize_dynamic(
    model=m, qconfig_spec={nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=False
)
"""


# # eager graph + static
# # https://github.com/Laicheng0830/Pytorch_Model_Quantization/blob/main/pose_estimation.py
# # 执行模式
# model.eval()
# # 获取默认的准备config阶段
# model_fp32_config = torch.quantization.get_default_qconfig('x86')
# # prepared 模型阶段
# model_fp32_prepared = torch.quantization.prepare(model)
# # 小范围推理
# evaluate(model_fp32_prepared)
# # 保存
# model_int8 = torch.quantization.convert(model_fp32_prepared)


# # FX + static
# from torch.ao.quantization import get_default_qconfig
# from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
# from torch.ao.quantization import QConfigMapping

# # z执行推理
# float_model.eval()

# # config
# qconfig = get_default_qconfig("x86")
# qconfig_mapping = QConfigMapping().set_global(qconfig)


# def calib(model, data_loader):
#     model.eval()
#     # 不累计梯度
#     with torch.no_grad():
#         for image, lable in data_loader:
#             model(image)


# example_inputs = next(iter(data_loader))[0]
# # prepared model
# prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)
# # 推理校准
# calibrate(prepared_model, data_loader_test)
# # quant
# quantized_model = convert_fx(prepared_model)


""" int8 量化
"""

# TODO 实际下划线的修改代表着原地修改数据的意思


# clamp_ 阶段误差
# round 近似误差
def quant_per_tensor_absmax(x, n_bit=8):
    scales = x.abs().max()
    q_max = 2 ** (n_bit - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    q_x = x / scales
    q_x = q_x.clamp_(-q_max, q_max).round_()
    return q_x, scales


def dequant(q_x, scales):
    return q_x * scales


X = torch.rand(2, 3, dtype=torch.float32)
W = torch.rand(3, 4, dtype=torch.float32)

# print(X)
# print(X.shape)

# print(W)
# print(W.shape)

q_x, x_scale = quant_per_tensor_absmax(X)
q_w, w_scale = quant_per_tensor_absmax(W)
q_y = torch.matmul(q_x, q_w)

Y_head = dequant(q_y, x_scale * w_scale)

Y = torch.matmul(X, W)
print(Y)

print(Y_head)
