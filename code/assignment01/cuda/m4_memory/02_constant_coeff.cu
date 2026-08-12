// 问题 4.3：把系数表搬进 constant memory（ MODIFY ）。
// 下面的 poly_eval_global 把 8 个多项式系数放在 global memory，每个线程读 8 次。
// 它保留不动，作为对比基准。
// 任务：
//   1. 声明 __constant__ float COEF[8]；
//   2. 在 main 里标了 TODO 的地方用 cudaMemcpyToSymbol 把系数拷进去；
//   3. 把 poly_eval_const 写成读 COEF 的版本——参数表保持不变（判测代码要用
//      同一个函数指针类型跑两版），里面不再用 coef 这个指针即可。
// 两版都要 PASS。评测结果会包含两版的耗时和比值。
#include "common.h"

__global__ void poly_eval_global(const float *x, float *y, const float *coef,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float acc = 0.f;
        // 秦九韶（Horner）算法，从最高次往下算。
        for (int k = 7; k >= 0; k--) acc = acc * xi + coef[k];
        y[i] = acc;
    }
}

__global__ void poly_eval_const(const float *x, float *y, const float *coef,
                                int n) {
    // TODO：从这里开始写（读 __constant__ COEF 的版本）
}

// ---------------- 以下是判测与计时，不要修改 ----------------

typedef void (*poly_fn)(const float *, float *, const float *, int);

static float run_one(poly_fn fn, const char *name, const float *d_x, float *d_y,
                     const float *d_coef, float *h_y, const float *h_ref, int n,
                     int blocks, int threads) {
    CUDA_CHECK(cudaMemset(d_y, 0, (size_t)n * sizeof(float)));
    fn<<<blocks, threads>>>(d_x, d_y, d_coef, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (!check_close(h_y, h_ref, n, 1e-3f)) {
        printf("%s: FAIL\n", name);
        emit_result("4.3", "fail", "{}");
        exit(1);
    }

    const int reps = 100;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < reps; r++) fn<<<blocks, threads>>>(d_x, d_y, d_coef, n);
    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();
    printf("%s: PASS  平均 %.4f ms\n", name, ms);
    return ms;
}

int main() {
    const int n = 1 << 24;
    size_t bytes = (size_t)n * sizeof(float);
    float h_coef[8] = {1.f, -0.5f, 0.25f, -0.125f, 0.0625f, -0.03125f, 0.015625f, -0.0078125f};

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_x, n, 5);
    for (int i = 0; i < n; i++) h_x[i] = h_x[i] * 0.1f;  // 压到 [0,1) 附近防溢出
    for (int i = 0; i < n; i++) {
        float acc = 0.f;
        for (int k = 7; k >= 0; k--) acc = acc * h_x[i] + h_coef[k];
        h_ref[i] = acc;
    }

    float *d_x, *d_y, *d_coef;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_coef, sizeof(h_coef)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coef, h_coef, sizeof(h_coef), cudaMemcpyHostToDevice));

    // TODO：把 h_coef 拷进你声明的 __constant__ 数组（cudaMemcpyToSymbol）。

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float ms_g = run_one(poly_eval_global, "global  ", d_x, d_y, d_coef, h_y,
                         h_ref, n, blocks, threads);
    float ms_c = run_one(poly_eval_const, "constant", d_x, d_y, d_coef, h_y,
                         h_ref, n, blocks, threads);
    // 这道题不预期提速，比值接近 1.00x 是正常结果，所以不设退化提示。
    float ratio = report_speedup("global / constant", ms_g, ms_c, 0.f, NULL);

    char metrics[192];
    snprintf(metrics, sizeof(metrics),
             "{\"global_ms\":%.4f,\"const_ms\":%.4f,\"speedup\":%.3f}", ms_g,
             ms_c, ratio);
    emit_result("4.3", "pass", metrics);
    return 0;
}
