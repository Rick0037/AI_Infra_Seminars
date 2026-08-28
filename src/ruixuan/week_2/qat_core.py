from turtle import forward

import torch
import torch.nn as nn


class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32: nn.Module) -> None:
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


quantized_model = QuantizedResNet18(model_fp32=fused_model)
# TODO 配置中给weight 以及 active 都规定好了，哪些叫使用，哪些叫没使用
quantization_config = torch.quantization.get_default_qconfig("x86")
quantized_model.qconfig = quantization_config
# https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
torch.quantization.prepare_qat(quantized_model, inplace=True)
# TODO inplace 实际上都是在某某地修改的模式
quantized_model.train()

# 这里不用 quantized_model 接也可以，因为使用了 inplace
quantized_model = torch.quantization.convert(quantized_model, inplace=True)

quantized_model.eval()

# 测试单步推理的耗时，实际上是以python 执行model的前后进行速度测试来进行的
# 直接单步的 quantized_model 开始测试了
