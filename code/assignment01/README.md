# Assignment 01 · GPU & GPU Programming

Session 1（SIMD/SIMT、CUDA 编程模型、Triton/TileLang）的配套作业。具体题目见 [assignment01.pdf](assignment01.pdf)。

## 目录结构

```
assignment01.pdf      作业本体 START HERE
cuda/                 模块 0-5、8 的 CUDA 练习
  common.h            错误检查、计时、对拍的公共工具
  Makefile            编译
  m2_first_kernel/judge_saxpy.sh   压轴题 2.9 的评测脚本（自带对拍）
  bonus/matmul.cu     Bonus 的 naive CUDA matmul
kernels/              模块 7 的 Python 练习（另含模块 1 的选做题、Bonus 的 Triton/TileLang matmul）
tests/                pytest 判测
```

## CUDA 部分

在有 NVIDIA GPU 和 CUDA Toolkit（建议 12.x）的机器上完成，集群的登录方式和环境以群内通知为准。

```bash
cd cuda
make run/m0_env/01_hello             # 编译并运行单个练习
ARCH=sm_80 make bin/...              # 集群上编译节点无卡时手动指定架构
```

## Python 部分

依赖（[uv](https://docs.astral.sh/uv/) 或 pip）：

```bash
uv sync --extra tilelang && uv run pytest tests/test_vector_add.py        # uv
python3 -m venv .venv && .venv/bin/pip install -e '.[tilelang]' && .venv/bin/pytest tests/test_vector_add.py   # pip
```

如没有 GPU，Triton 相关题目也可在本地通过 `TRITON_INTERPRET=1` 过正确性测试。但 interpreter 模式下的耗时没有参考价值，性能相关的选做内容要在 GPU 上跑。

## 所有题目完成判测

```bash
cd cuda && for f in bin/m*/*; do case $f in *_sassonly|*_ptxonly) continue;; esac; echo "== $f"; $f; done  # CUDA 各题（跳过预期报错的 case）
pytest tests/                                          # Python 各题
./m2_first_kernel/judge_saxpy.sh path/to/saxpy.cu      # 压轴题 2.9
```
