"""问题 7.3：TileLang 版 scale-add（填空）。

Y = 2 * X + 1，X 形状 (M, N)。两个空对应 TileLang 的两个 basic operation。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k scale_add
"""

import tilelang
import tilelang.language as T


def make_scale_add(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale_add(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # ====== 空 1：二维 CTA grid——x 方向要多少个 block（管 N 列），
        #         y 方向要多少个（管 M 行）？提示：T.ceildiv ======
        with T.Kernel(..., ..., threads=128) as (bx, by):
            # ====== 空 2：block 内并行遍历 tile 的每个元素，
            #         提示：T.Parallel(维度1, 维度2) ======
            for i, j in ...:
                gi = by * block_M + i
                gj = bx * block_N + j
                if gi < M and gj < N:
                    Y[gi, gj] = X[gi, gj] * 2.0 + 1.0

    return scale_add
