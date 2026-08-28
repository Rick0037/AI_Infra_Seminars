import torch
import torch.nn as nn


"""
【编码题】用 Python (numpy或者pytorch) 实现 symmetric vs asymmetric INT8 量化数学过程，为后续推进课程五量化部分打基础。要求：
1. 实现两个函数 — symmetric quant: scale = abs_max/127, q = round(x/scale) clip [-128,127]；
asymmetric quant: scale = (max-min)/255, zp = round(-min/scale), q = round(x/scale + zp) clip [0,255]，并写对应 dequant；
2. 构造三种输入分布：均匀 U(-1,1)、正态 N(0,1)、N(0,0.1) 注入 1% outlier (值域 [-5,5])；
3. 对每种分布跑 per-tensor 量化 + 反量化，输出 MSE 误差表；
4. 得出结论：哪种分布下 asymmetric 比 symmetric 优势明显？为什么？
"""


class SymmetricQuant:
    def __init__(self, n_bit) -> None:
        self.n_bit = n_bit

    # return 数值是 scale 以及量化后的 x_quant
    def Quant(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_quant_max = 2 ** (self.n_bit - 1) - 1
        x_real_max = x.abs().max()
        scale = x_real_max / x_quant_max
        # 预防 scale 为0
        scale.clamp_(min=1e-5)
        x_quant = x.div(scale)
        # 四舍五入
        x_quant.round_()
        x_quant.clamp_(-x_quant_max, x_quant_max)
        return x_quant.to(torch.int8), scale

    def Dequant(self, x_quant: torch.Tensor, scale) -> torch.Tensor:
        return x_quant.float() * scale


# 这里不是非对称，这里实际上就是uint8的层级
# quant 函数都需要 round 之后进行clamp
class AsymmetricQuant:
    def __init__(self, n_bit) -> None:
        self.n_bit = n_bit

    def Quant(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_quant_max = torch.pow(2, self.n_bit - 1) - 1
        # x_quant_min = -torch.pow(2, self.n_bit - 1)
        x_quant_max = 2**self.n_bit - 1
        x_quant_min = 0
        x_max = x.max()
        x_min = x.min()
        scale = (x_max - x_min) / (x_quant_max - x_quant_min)
        scale.clamp_(min=1e-5)
        # (x-x_min) / (zp -x_quant_min) = scale
        zero_pt = (-x_min).div(scale).round_()
        x_quant = (x.div(scale) + zero_pt).round_()
        x_quant.clamp_(x_quant_min, x_quant_max)
        # 为什么 要 clip [0,255] 因为：如果输入 0 不在 [min, max] 范围内（比如上面的 [3, 10]），
        # 算出来的 q 会超出 [0, 255]（如 q=-109），所以必须 clip [0, 255] 防止溢出 uint8 范围
        return x_quant.to(torch.uint8), scale, zero_pt

    def Dequant(self, x: torch.Tensor, scale, zero_pt) -> torch.Tensor:
        return (x.float() - zero_pt) * scale


def get_distribution(name: str, number: int, outlier: float = 0.01):
    shape = (number,)
    if name == "uniform":
        x = 2 * torch.rand(shape) - 1.0
    elif name == "normal":
        x = torch.randn(shape)
    elif name == "normal_01":
        x = torch.randn(shape) * 0.1
        k = int(number * outlier)
        index = torch.randperm(number)[:k]
        x[index] = torch.rand(k) * 10 - 5
    return x


def MSE(x, fake_x):
    return torch.mean((x - fake_x) ** 2)


def main():
    syme = SymmetricQuant(8)
    asyme = AsymmetricQuant(8)

    for name in ["uniform", "normal", "normal_01"]:
        x = get_distribution(name, 2000)
        x_quant, scale = syme.Quant(x)
        x_fake = syme.Dequant(x_quant, scale)
        x_syme_mse = MSE(x, x_fake)

        x_aquant, scale, zpt = asyme.Quant(x)
        x_afake = asyme.Dequant(x_aquant, scale, zpt)
        x_asyme_mse = MSE(x, x_afake)

        print(f"{name}: sym MSE={x_syme_mse:.6f}, asym MSE={x_asyme_mse:.6f}")


if __name__ == "__main__":
    main()
