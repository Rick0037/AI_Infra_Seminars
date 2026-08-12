// 问题 3.2：divergence
// 两个 kernel 的每个线程做同样多的计算，区别只在分支怎么划分线程。
// 先在 handout 上写下你的预测，再运行对比。
#include "common.h"

// 按奇偶分支：同一个 warp 里两种线程各占一半。
__global__ void diverge_in_warp(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 0.5f;
    if (tid % 2 == 0) {
        for (int i = 0; i < iters; i++) x = x * 1.000001f + 0.5f;
    } else {
        for (int i = 0; i < iters; i++) x = x * 0.999999f - 0.5f;
    }
    out[tid] = x;
}

// 按 warp 分支：一个 warp 内所有线程走同一条路。
__global__ void diverge_by_warp(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 0.5f;
    if ((tid / 32) % 2 == 0) {
        for (int i = 0; i < iters; i++) x = x * 1.000001f + 0.5f;
    } else {
        for (int i = 0; i < iters; i++) x = x * 0.999999f - 0.5f;
    }
    out[tid] = x;
}

int main() {
    const int blocks = 1024, threads = 256, iters = 20000;
    const int n = blocks * threads;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));

    // 各热身一次。
    diverge_in_warp<<<blocks, threads>>>(d_out, iters);
    diverge_by_warp<<<blocks, threads>>>(d_out, iters);
    CUDA_CHECK_KERNEL();

    GpuTimer timer;

    timer.start();
    diverge_in_warp<<<blocks, threads>>>(d_out, iters);
    float ms_in = timer.stop_ms();

    timer.start();
    diverge_by_warp<<<blocks, threads>>>(d_out, iters);
    float ms_by = timer.stop_ms();

    CUDA_CHECK_KERNEL();
    printf("warp 内分支 (tid %% 2)    : %8.3f ms\n", ms_in);
    printf("按 warp 分支 (tid/32 %% 2): %8.3f ms\n", ms_by);
    printf("比值: %.2f\n", ms_in / ms_by);
    return 0;
}
