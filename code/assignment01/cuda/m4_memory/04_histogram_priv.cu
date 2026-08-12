// 问题 4.6：直方图私有化（ MODIFY ）。
// 下面的 histogram_naive 是问题 4.5 的成品：所有线程都在挤同一组全局计数器
// （256 个 bucket）。它保留不动，作为对比基准。
// 任务：把 histogram_priv 写成 shared memory 私有化版——
//   1. 每个 block 在 shared memory 里声明自己的计数器并清零；
//   2. block 内线程往自己的计数器里 atomicAdd；
//   3. 同步之后，把 shared 直方图的 256 个 bucket 用 atomicAdd 汇入全局直方图。
// 两版都要 PASS。评测结果会包含两版的耗时和比值，解释提速来自哪里。
#include "common.h"

#define BINS 256

__global__ void histogram_naive(const unsigned char *data, unsigned int *hist,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        atomicAdd(&hist[data[i]], 1u);
    }
}

__global__ void histogram_priv(const unsigned char *data, unsigned int *hist,
                               int n) {
    // TODO：从这里开始写（shared memory 私有化版本）
}

// ---------------- 以下是判测与计时，不要修改 ----------------

typedef void (*hist_fn)(const unsigned char *, unsigned int *, int);

static float run_one(hist_fn fn, const char *name, const unsigned char *d_data,
                     unsigned int *d_hist, const unsigned int *h_ref, int n,
                     int blocks, int threads) {
    unsigned int h_hist[BINS];
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
    fn<<<blocks, threads>>>(d_data, d_hist, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    for (int b = 0; b < BINS; b++) {
        if (h_hist[b] != h_ref[b]) {
            fprintf(stderr, "bin %d: got %u, want %u\n", b, h_hist[b], h_ref[b]);
            printf("%s: FAIL\n", name);
            emit_result("4.6", "fail", "{}");
            exit(1);
        }
    }

    const int reps = 50;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < reps; r++) fn<<<blocks, threads>>>(d_data, d_hist, n);
    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();
    printf("%s: PASS  平均 %.4f ms  (%.2f GB/s)\n", name, ms, n / ms / 1e6);
    return ms;
}

int main() {
    const int n = 1 << 24;

    unsigned char *h_data = (unsigned char *)malloc(n);
    unsigned int h_ref[BINS] = {0};
    srand(9);
    for (int i = 0; i < n; i++) h_data[i] = (unsigned char)(rand() % BINS);
    for (int i = 0; i < n; i++) h_ref[h_data[i]]++;

    unsigned char *d_data;
    unsigned int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));

    int threads = 256, blocks = 1024;
    float ms_naive = run_one(histogram_naive, "naive", d_data, d_hist, h_ref, n,
                             blocks, threads);
    float ms_priv = run_one(histogram_priv, "priv ", d_data, d_hist, h_ref, n,
                            blocks, threads);
    // 阈值 10x：A100 实测 149x、V100 实测 86x，失败信号（没真私有化）是 ~1x。
    float ratio = report_speedup("naive / priv", ms_naive, ms_priv, 10.0f,
                                 "提速不到 10x，检查私有化是不是真的生效了");

    // 私有化版实际用了多少 shared memory——只报数，不作为判定条件。
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, histogram_priv));
    if (attr.sharedSizeBytes == 0) {
        printf("WARN: priv 版没有用到 shared memory（不影响 PASS）\n");
    }

    char metrics[256];
    snprintf(metrics, sizeof(metrics),
             "{\"naive_ms\":%.4f,\"priv_ms\":%.4f,\"speedup\":%.3f,"
             "\"shared_bytes\":%zu}",
             ms_naive, ms_priv, ratio, attr.sharedSizeBytes);
    emit_result("4.6", "pass", metrics);
    return 0;
}
