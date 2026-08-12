#!/usr/bin/env bash
# 问题 2.9（压轴）的评测脚本，自带对拍。
# 用法：./judge_saxpy.sh [你的源文件，默认 saxpy.cu]
# 环境变量：ARCH=sm_80 指定目标架构（默认 native，即当前这块卡；编译节点无卡时要指定）
#           WMHPC_RESULT=1 额外输出一行机器可读的 ##RESULT
# contract 见 handout：./saxpy <n> 会按固定公式生成 x、y，在 GPU 上算 y = 2x + y，
# 输出一行 SUM=<所有 y[i] 之和>，退出码 0。
# 脚本用同样的公式在 CPU 上独立算一遍期望值，和你的 SUM 对照——
# 公式生成的值都是较小的整数或半整数，float 下能精确表示，直接比较整数值即可。
set -u
SRC="${1:-saxpy.cu}"
ARCH="${ARCH:-native}"
BIN="$(mktemp -u /tmp/saxpy.XXXXXX)"

emit_result() {  # $1=status $2=metrics_json
    [[ "${WMHPC_RESULT:-0}" == "0" || -z "${WMHPC_RESULT:-}" ]] && return 0
    local dev
    dev="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    printf '##RESULT {"prob":"2.9","status":"%s","metrics":%s,"device":"%s"}\n' \
        "$1" "$2" "${dev:-unknown}"
}

if [[ ! -f "$SRC" ]]; then
    echo "找不到 $SRC"
    emit_result "not_attempted" '{}'
    exit 1
fi

if ! nvcc -O2 -std=c++17 -arch="$ARCH" -o "$BIN" "$SRC"; then
    echo "编译失败"
    emit_result "compile_error" '{}'
    exit 1
fi

expected_sum() {
    awk -v n="$1" 'BEGIN {
        s = 0;
        for (i = 0; i < n; i++) {
            x = (i % 2048 - 1024) * 0.5;
            y = i % 1024 - 512;
            s += 2 * x + y;
        }
        printf "%.0f", s;
    }'
}

fail=0
npass=0
ntotal=0
for n in 0 1 31 1024 1025 1048576 1048579; do
    ntotal=$((ntotal + 1))
    out="$("$BIN" "$n" 2>&1)"
    rc=$?
    got="$(printf '%s\n' "$out" | grep -o 'SUM=[-0-9]*' | tail -1 | cut -d= -f2)"
    want="$(expected_sum "$n")"
    if [[ $rc -eq 0 && -n "$got" && "$got" == "$want" ]]; then
        echo "n=$n  PASS  (SUM=$got)"
        npass=$((npass + 1))
    else
        echo "n=$n  FAIL  (期望 SUM=$want，退出码 $rc，你的输出如下)"
        printf '%s\n' "$out" | head -5
        fail=1
    fi
done
rm -f "$BIN"

metrics="$(printf '{"cases_total":%d,"cases_passed":%d}' "$ntotal" "$npass")"
if [[ $fail -eq 0 ]]; then
    echo "全部通过"
    emit_result "pass" "$metrics"
else
    emit_result "fail" "$metrics"
    exit 1
fi
