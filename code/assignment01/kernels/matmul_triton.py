"""Bonus（选做）：Triton tiled matmul。

任务：调 bench() 的 BLOCK_M / BLOCK_N / BLOCK_K，记录实测数据。
    哪组参数最快？差多少？为什么 tile 大小会影响这么大

    python -c "from kernels.matmul_triton import bench; bench()"
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor,
           BLOCK_M=64, BLOCK_N=64, BLOCK_K=32) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def bench(M=2048, N=2048, K=2048):
    assert torch.cuda.is_available(), "benchmark 需要 GPU"
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    configs = [(32, 32, 32), (64, 64, 32), (128, 64, 32), (128, 128, 32),
               (128, 128, 64), (64, 64, 64)]
    for bm, bn, bk in configs:
        ms = triton.testing.do_bench(lambda: matmul(a, b, bm, bn, bk))
        tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
        print(f"BLOCK_M={bm:4d} BLOCK_N={bn:4d} BLOCK_K={bk:3d}  "
              f"{ms:8.3f} ms  {tflops:6.1f} TFLOPS")
    ms = triton.testing.do_bench(lambda: a @ b)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    print(f"torch (cuBLAS)                      {ms:8.3f} ms  {tflops:6.1f} TFLOPS")
