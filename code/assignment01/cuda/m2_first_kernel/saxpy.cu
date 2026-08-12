// 问题 2.9（压轴）：saxpy —— y = 2*x + y（单精度）
// 不依赖 common.h，错误检查宏和 cudaEvent 计时都自己写一遍。
// 用法：./saxpy <n>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 包住每个 CUDA API 调用，出错立刻报出文件、行号和原因。
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                    cudaGetErrorName(err_), __FILE__, __LINE__,            \
                    cudaGetErrorString(err_));                             \
            exit(1);                                                       \
        }                                                                  \
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

// saxpy kernel：每个线程处理一个元素 y[i] = 2*x[i] + y[i]
__global__ void saxpy_kernel(const float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 2.0f * x[idx] + y[idx];
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "用法: %s <n>\n", argv[0]);
        return 1;
    }
    long n = atol(argv[1]);

    // 特判 n=0：0 个 block 的 kernel launch 非法，直接输出 SUM=0
    if (n == 0) {
        printf("SUM=0 n=0 ms=0\n");
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);

    // host 分配
    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    if (!h_x || !h_y) {
        fprintf(stderr, "malloc 失败\n");
        return 1;
    }

    // 按固定公式生成数据（都是 float）
    for (long i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (float)((i % 1024) - 512);
    }

    // device 分配
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // 拷贝到 device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // 启动 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((n + threadsPerBlock - 1) / threadsPerBlock);

    GpuTimer timer;
    timer.start();
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, (int)n);
    // kernel 启动本身没有返回值，要靠这两句查它的错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = timer.stop_ms();

    // 拷回 host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // 用 double 累加所有 y[i]
    double s = 0.0;
    for (long i = 0; i < n; i++) {
        s += (double)h_y[i];
    }

    // 输出：SUM 用 %.0f 保证只输出整数（值都是整数，无小数点）
    printf("SUM=%.0f n=%ld ms=%.3f\n", s, n, ms);

    // 释放资源
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}
