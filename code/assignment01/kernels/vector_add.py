"""问题 7.1：Triton 向量加法（填空）。

四个空对应 Triton kernel 的四个 basic operation。填完运行：
    pytest tests/test_vector_add.py
没有 GPU 也能跑，conftest.py 会自动切到 interpreter 模式。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    # ====== 空 1：当前 program 在一维 grid 里的编号 ======
    pid = ...
    # ====== 空 2：这个 program 负责的一段全局下标（长度 BLOCK_SIZE） ======
    offsets = ...
    # ====== 空 3：屏蔽越界位置的 mask ======
    mask = ...

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # ====== 空 4：把 x + y 写回 z（别忘了 mask） ======
    ...


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z
