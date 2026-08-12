from kernels.simt_sim import run


def test_uniform():
    regs, cycles = run([("add", 1), ("mul", 2)])
    assert regs == [(i + 1) * 2 for i in range(32)]
    assert cycles == 2


def test_divergent_both_sides():
    regs, cycles = run([("if_lt", 16, [("add", 100)], [("add", 200)])])
    assert regs == [i + 100 if i < 16 else i + 200 for i in range(32)]
    # 两个分支各执行一遍：1 + 1 拍。
    assert cycles == 2


def test_one_sided_branch_skips():
    regs, cycles = run([("if_lt", 32, [("add", 1)], [("add", 5)])])
    assert regs == [i + 1 for i in range(32)]
    # else 分支没有 active lane，整支跳过、不计拍。
    assert cycles == 1


def test_nested():
    prog = [
        ("if_lt", 16,
         [("if_lt", 8, [("add", 1)], [("add", 2)])],
         [("add", 3)]),
    ]
    regs, cycles = run(prog)
    want = []
    for i in range(32):
        if i < 8:
            want.append(i + 1)
        elif i < 16:
            want.append(i + 2)
        else:
            want.append(i + 3)
    assert regs == want
    assert cycles == 3


def test_after_reconverge():
    prog = [
        ("if_lt", 16, [("add", 10)], [("add", 20)]),
        ("mul", 2),  # 汇合之后全体执行，只算 1 拍
    ]
    regs, cycles = run(prog)
    assert regs == [(i + 10) * 2 if i < 16 else (i + 20) * 2 for i in range(32)]
    assert cycles == 3
