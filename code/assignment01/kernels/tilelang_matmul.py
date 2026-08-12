"""问题 7.6：TileLang tiled matmul（填空）。

C = A @ B，A 形状 (M, K)，B 形状 (K, N)，fp16 输入、fp32 累加。
共五个空：两块 shared tile、一个 fragment 累加器、沿 K 维的流水
循环、T.copy 搬运与 T.gemm 计算。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k matmul

Bonus 用的也是这个文件：填完后调 bench() 里的配置，记录实测数据。

    python -c "from kernels.tilelang_matmul import bench; bench()"
"""

import tilelang
import tilelang.language as T


def make_matmul(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                threads=128, num_stages=3,
                dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, BLOCK_N),
            T.ceildiv(M, BLOCK_M),
            threads=threads,
        ) as (bx, by):
            # ====== 空 1：A、B 各自的 shared tile——形状分别是多少？ ======
            A_shared = T.alloc_shared(..., dtype)
            B_shared = T.alloc_shared(..., dtype)

            # ====== 空 2：C 的累加器 tile，放寄存器（fragment），
            #         注意精度用 accum_dtype ======
            C_local = T.alloc_fragment(..., accum_dtype)

            T.clear(C_local)

            # ====== 空 3：沿 K 维流水地推进——一共要多少步？提示：T.ceildiv ======
            for k in T.Pipelined(..., num_stages=num_stages):
                # ====== 空 4：把 A、B 的当前 tile 搬进 shared——
                #         各自的全局起点坐标是多少？ ======
                T.copy(A[..., ...], A_shared)
                T.copy(B[..., ...], B_shared)
                # ====== 空 5：tile 级乘累加，提示：T.gemm ======
                ...

            T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    return main


def bench(M=2048, N=2048, K=2048):
    import torch
    import triton

    assert torch.cuda.is_available(), "benchmark 需要 GPU"
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # =====就在这里修改，可以加入多条config
    # (block_M, block_N, block_K, threads, num_stages)
    configs = [
        (64, 64, 32, 128, 1),
        (64, 64, 32, 128, 3),
        (128, 128, 32, 128, 3),
        (128, 128, 64, 256, 3),
        (128, 256, 64, 256, 3),
    ]

    checked = False
    for bm, bn, bk, threads, stages in configs:
        kernel = tilelang.compile(
            make_matmul(M, N, K, bm, bn, bk, threads=threads, num_stages=stages),
            out_idx=[2],
        )
        if not checked:  # 第一个配置和 torch 对拍一次
            ref = a.float() @ b.float()
            torch.testing.assert_close(kernel(a, b), ref, rtol=1e-2, atol=1e-1)
            checked = True
        ms = triton.testing.do_bench(lambda: kernel(a, b))
        tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
        print(f"block_M={bm:4d} block_N={bn:4d} block_K={bk:3d} "
              f"threads={threads:3d} stages={stages}  {ms:8.3f} ms  {tflops:6.1f} TFLOPS")
