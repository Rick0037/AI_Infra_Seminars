import os

import pytest

# 评测模式。外部评测框架跑判测时设 WMHPC_GRADING=1，自己做题时不要设。
# 它做两件事：拒绝在没有 GPU 的机器上跑，以及把「跳过」按「未完成」计——
# 否则没装 tilelang 或没有卡的机器会一片绿，一行代码没写也是满分。
GRADING = os.environ.get("WMHPC_GRADING", "0") not in ("0", "")

if GRADING:
    # 评测模式下绝不退化到 interpreter。
    os.environ.pop("TRITON_INTERPRET", None)
elif "TRITON_INTERPRET" not in os.environ:
    # 没有 GPU 时自动切换到 Triton interpreter 模式（纯 CPU）。
    # interpreter 只验证正确性，性能数字没有参考价值。
    # 这段必须在任何测试 import triton 之前执行，pytest 加载 conftest 的
    # 时机恰好满足这一点。
    try:
        import torch

        if not torch.cuda.is_available():
            os.environ["TRITON_INTERPRET"] = "1"
    except ImportError:
        pass


def pytest_configure(config):
    if GRADING:
        # 收集阶段的 skip 被改判成 failed 会中断整场收集，
        # 这里放行，让其余题目照常判测。
        config.option.continue_on_collection_errors = True


def pytest_sessionstart(session):
    if not GRADING:
        return
    try:
        import torch
    except ImportError:
        raise pytest.UsageError("WMHPC_GRADING=1 但没装 torch。")
    if not torch.cuda.is_available():
        raise pytest.UsageError(
            "WMHPC_GRADING=1 但这台机器没有可用的 CUDA 设备。评测必须在有卡的"
            "节点上跑，否则 Triton 题会走 interpreter 假装通过。"
        )


def _skip_is_failure(report):
    """评测模式下把 skipped 改判为 failed，并保留原因。"""
    if not GRADING or report.outcome != "skipped":
        return
    reason = report.longrepr
    if isinstance(reason, tuple) and len(reason) == 3:
        reason = reason[2]
    report.outcome = "failed"
    report.longrepr = f"WMHPC_GRADING=1：跳过按未完成计（原因：{reason}）"


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    report = yield
    _skip_is_failure(report)
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    # 模块级 skip（pytest.importorskip、pytest.skip(allow_module_level=True)）
    # 发生在收集阶段，走的是这条路，不是上面那个钩子。
    report = yield
    _skip_is_failure(report)
    return report
