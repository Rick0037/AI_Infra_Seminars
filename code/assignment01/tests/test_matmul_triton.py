import torch

from kernels.matmul_triton import matmul


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# 输入用 fp16：GPU 上 fp32 输入的 tl.dot 默认走 TF32（10 位尾数），
# 数值和 CPU 参考对不上；fp16 输入 + fp32 累加在两种后端行为一致。
def _check(M, K, N, seed):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device=_device(), dtype=torch.float16)
    b = torch.randn(K, N, device=_device(), dtype=torch.float16)
    got = matmul(a, b, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16)
    ref = a.float() @ b.float()
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)


def test_matmul_small():
    _check(64, 48, 32, seed=0)


def test_matmul_ragged():
    _check(100, 70, 55, seed=1)
