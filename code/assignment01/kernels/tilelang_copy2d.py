"""问题 7.4：TileLang 版二维缩放拷贝（填空）。

Y = 2 * X，X 形状 (M, N)，M、N 都不保证整除 tile 边长 (类似 prob 2.6 )
。这次把 tile 先搬进 shared memory，算完再写回：数据搬运交给
T.copy.
填完对照 2.6 想一想：行列号、边界保护、grid 尺寸这几个空，
哪些在这里还有对应，哪些被 T.copy 吃掉了。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k copy2d
"""

import tilelang
import tilelang.language as T


def make_scale2d(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale2d(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # ====== 空 1：二维 CTA grid，和 7.3 一样——x 方向管 N 列，
        #         y 方向管 M 行，提示：T.ceildiv ======
        with T.Kernel(..., ..., threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), dtype)

            # ====== 空 2：把当前 tile 从 X 搬进 shared。
            #         提示：T.copy(X[行起点, 列起点], X_shared)，
            #         越界部分 T.copy 会自己处理 ======
            ...

            for i, j in T.Parallel(block_M, block_N):
                X_shared[i, j] = X_shared[i, j] * 2.0

            # ====== 空 3：把算完的 tile 写回 Y 的同一位置 ======
            ...

    return scale2d
