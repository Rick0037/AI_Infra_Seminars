"""问题 7.2：fused elementwise（改造题）。

scale_kernel 目前功能完整，相应代码不要变动。
fused_kernel 目前和 scale_kernel 完全一致，是你需要修改的 kernel。
任务：改成 z = relu(a * x + b)，其中 a、b 是标量。
TIP: 只需要动计算那一行，再把 a、b 传进 kernel——主体不变，
这正是 Tile 视角的好处:-)。改完运行：
    pytest tests/test_fused_op.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = x * 2.0
    tl.store(z_ptr + offsets, z, mask=mask)


def scale(x: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    scale_kernel[grid](x, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z


# ====== 从这里开始改 ======

@triton.jit
def fused_kernel(x_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = x * 2.0  # TODO：改成 relu(a * x + b)，提示 tl.maximum
    tl.store(z_ptr + offsets, z, mask=mask)


def fused(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    fused_kernel[grid](x, z, n, BLOCK_SIZE=BLOCK_SIZE)  # TODO：把 a、b 传进去
    return z
