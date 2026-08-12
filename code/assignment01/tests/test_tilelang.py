import pytest
import torch

tilelang = pytest.importorskip("tilelang", reason="需要安装 tilelang（uv sync --extra tilelang）")

if not torch.cuda.is_available():
    pytest.skip("TileLang 题需要 GPU，在集群上运行", allow_module_level=True)


def test_scale_add():
    from kernels.tilelang_scale_add import make_scale_add

    M, N = 123, 77
    func = make_scale_add(M, N)
    kernel = tilelang.compile(func, out_idx=[1])
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y = kernel(x)
    torch.testing.assert_close(y, x * 2.0 + 1.0, atol=1e-5, rtol=1e-5)


def test_copy2d():
    from kernels.tilelang_copy2d import make_scale2d

    for M, N in [(64, 128), (123, 77), (1, 500), (500, 1)]:
        func = make_scale2d(M, N)
        kernel = tilelang.compile(func, out_idx=[1])
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(kernel(x), x * 2.0)


def test_matmul():
    from kernels.tilelang_matmul import make_matmul

    M, N, K = 256, 256, 128
    func = make_matmul(M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    kernel = tilelang.compile(func, out_idx=[2])
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    ref = (a.float() @ b.float())
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
