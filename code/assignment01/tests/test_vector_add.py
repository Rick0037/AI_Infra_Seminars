import torch

from kernels.vector_add import add


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_exact_multiple():
    torch.manual_seed(0)
    x = torch.randn(4096, device=_device())
    y = torch.randn(4096, device=_device())
    torch.testing.assert_close(add(x, y), x + y)


def test_ragged():
    torch.manual_seed(1)
    x = torch.randn(10000, device=_device())
    y = torch.randn(10000, device=_device())
    torch.testing.assert_close(add(x, y), x + y)


def test_tiny():
    torch.manual_seed(2)
    x = torch.randn(3, device=_device())
    y = torch.randn(3, device=_device())
    torch.testing.assert_close(add(x, y), x + y)
