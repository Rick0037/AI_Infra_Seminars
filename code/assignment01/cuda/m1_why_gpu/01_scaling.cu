// 问题 1.5：并行度与吞吐。同一个向量加法，用四种方式跑一遍，
// 对比每种方式的耗时并解释差距。
// 编译运行：make run/m1_why_gpu/01_scaling
#include <chrono>
#include "common.h"

// 整块 GPU 上只有一个线程在干活。
__global__ void add_one_thread(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

// 一个 block、256 个线程。
__global__ void add_one_block(const float *a, const float *b, float *c, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) c[i] = a[i] + b[i];
}

// 铺满整个 grid，一个线程管一个元素。
__global__ void add_grid(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 22;  // 4M 元素
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);

    // CPU 单线程作基准。
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU 单线程      : %10.3f ms  (%6.2f ns/元素)\n", cpu_ms,
           cpu_ms * 1e6 / n);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 热身，避免首次启动的开销混进计时。
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_grid<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    GpuTimer timer;

    timer.start();
    add_one_thread<<<1, 1>>>(d_a, d_b, d_c, n);
    float ms1 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_c, h_ref, n)) REPORT(0);
    printf("GPU <<<1, 1>>>  : %10.3f ms  (%6.2f ns/元素)\n", ms1, ms1 * 1e6 / n);

    timer.start();
    add_one_block<<<1, 256>>>(d_a, d_b, d_c, n);
    float ms2 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_c, h_ref, n)) REPORT(0);
    printf("GPU <<<1, 256>>>: %10.3f ms  (%6.2f ns/元素)\n", ms2, ms2 * 1e6 / n);

    timer.start();
    add_grid<<<blocks, threads>>>(d_a, d_b, d_c, n);
    float ms3 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_c, h_ref, n)) REPORT(0);
    printf("GPU 铺满 grid   : %10.3f ms  (%6.2f ns/元素, %d blocks x %d threads)\n",
           ms3, ms3 * 1e6 / n, blocks, threads);

    REPORT(1);
    return 0;
}
