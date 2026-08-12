// 问题 5.1：计时陷阱。
// 同一个 kernel，三种计时方式给出三个数，判断哪个数能作为 performance index
// （另外两种具体测量的是什么时间？）
#include <chrono>
#include "common.h"

__global__ void busy(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 0.5f;
    for (int i = 0; i < iters; i++) x = x * 1.000001f + 0.5f;
    out[tid] = x;
}

int main() {
    const int blocks = 2048, threads = 256, iters = 5000;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)blocks * threads * sizeof(float)));

    busy<<<blocks, threads>>>(d_out, iters);  // 热身
    CUDA_CHECK_KERNEL();

    // 方式一：host 计时，启动后立刻停表。
    auto t0 = std::chrono::steady_clock::now();
    busy<<<blocks, threads>>>(d_out, iters);
    auto t1 = std::chrono::steady_clock::now();
    double ms_nosync = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CUDA_CHECK(cudaDeviceSynchronize());

    // 方式二：host 计时，等 GPU 干完再停表。
    t0 = std::chrono::steady_clock::now();
    busy<<<blocks, threads>>>(d_out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = std::chrono::steady_clock::now();
    double ms_sync = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 方式三：cudaEvent 计时。
    GpuTimer timer;
    timer.start();
    busy<<<blocks, threads>>>(d_out, iters);
    float ms_event = timer.stop_ms();

    printf("host 计时、不等 GPU : %10.4f ms\n", ms_nosync);
    printf("host 计时、等 GPU   : %10.4f ms\n", ms_sync);
    printf("cudaEvent 计时      : %10.4f ms\n", ms_event);
    return 0;
}
