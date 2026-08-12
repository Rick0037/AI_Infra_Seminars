import torch

from kernels.fused_op import fused, scale


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_scale_still_works():
    # 原有的 scale 不许改坏。
    torch.manual_seed(0)
    x = torch.randn(5000, device=_device())
    torch.testing.assert_close(scale(x), x * 2.0)


def test_fused():
    torch.manual_seed(1)
    x = torch.randn(10000, device=_device())
    a, b = 1.5, -0.3
    torch.testing.assert_close(fused(x, a, b), torch.relu(a * x + b))


def test_fused_other_params():
    torch.manual_seed(2)
    x = torch.randn(777, device=_device())
    a, b = -2.0, 0.7
    torch.testing.assert_close(fused(x, a, b), torch.relu(a * x + b))
