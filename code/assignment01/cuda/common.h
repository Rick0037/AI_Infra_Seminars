// 本目录所有练习共用的小工具。压轴题（问题 2.9）要求自己写一遍
// 这里的错误检查和计时，做到那题时别 include 这个文件。
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 包住每个 CUDA API 调用，出错立刻报出文件、行号和原因。
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// kernel 启动本身没有返回值，要靠这两句查它的错误。
#define CUDA_CHECK_KERNEL()                        \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)

// 基于 cudaEvent 的计时器，量的是 GPU 上的耗时（毫秒）。
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// 固定种子的伪随机填充，保证每次运行数据一致。
static inline void fill_random(float *p, long n, unsigned seed = 42) {
    srand(seed);
    for (long i = 0; i < n; i++) p[i] = (float)(rand() % 1000) / 100.0f;
}

// 逐元素对拍，相对误差超过 eps 视为失败，返回 1 表示通过。
static inline int check_close(const float *got, const float *want, long n,
                              float eps = 1e-4f) {
    for (long i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > eps * (1.0f + fabsf(want[i]))) {
            fprintf(stderr, "MISMATCH at %ld: got %f, want %f\n", i,
                    (double)got[i], (double)want[i]);
            return 0;
        }
    }
    return 1;
}

#define REPORT(ok)                       \
    do {                                 \
        if (ok) {                        \
            printf("PASS\n");            \
        } else {                         \
            printf("FAIL\n");            \
            exit(1);                     \
        }                                \
    } while (0)

// ---------------- 性能对比 ----------------
// 打印两版的比值。ratio 低于 warn_below 时多打一句提示，但不改变退出码——
// 判测只负责把数字摆出来，快慢的判断留给你自己和 handout 上的解释。
// warn_below <= 0 表示这道题本来就不预期提速，不打提示。
static inline float report_speedup(const char *label, float base_ms,
                                   float opt_ms, float warn_below,
                                   const char *hint) {
    float ratio = opt_ms > 0.f ? base_ms / opt_ms : 0.f;
    printf("%s = %.2fx\n", label, ratio);
    if (warn_below > 0.f && ratio < warn_below) {
        printf("WARN: %s（不影响 PASS）\n", hint);
    }
    return ratio;
}

// ---------------- 机器可读的结果行 ----------------
// 供外部评测框架消费。默认不打，
// 设了环境变量 WMHPC_RESULT=1 才输出，日常运行的输出保持干净。
// 约定：退出码只表达正确性，性能数字永远不影响退出码。
static inline void emit_result(const char *prob, const char *status,
                               const char *metrics_json) {
    const char *on = getenv("WMHPC_RESULT");
    if (!on || on[0] == '0' || on[0] == '\0') return;
    if (!metrics_json) metrics_json = "{}";

    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        printf("##RESULT {\"prob\":\"%s\",\"status\":\"%s\",\"metrics\":%s,"
               "\"device\":\"%s\",\"sm\":%d,\"cc\":\"%d.%d\"}\n",
               prob, status, metrics_json, prop.name,
               prop.multiProcessorCount, prop.major, prop.minor);
    } else {
        printf("##RESULT {\"prob\":\"%s\",\"status\":\"%s\",\"metrics\":%s}\n",
               prob, status, metrics_json);
    }
    fflush(stdout);
}
