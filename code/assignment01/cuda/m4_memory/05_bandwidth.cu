// 问题 4.7：访存模式与带宽。
// 同一个 kernel，只改读取的步长。
#include "common.h"

// stride = 1 时是连续访问；stride 变大后，warp 里相邻线程读的地址
// 相距 stride 个 float。n 是 2 的幂，& (n-1) 等价于取模。
__global__ void strided_copy(const float *in, float *out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (long)i * stride & (n - 1);
        out[i] = in[j];
    }
}

int main() {
    const int n = 1 << 24;  // 16M 元素，2 的幂
    size_t bytes = (size_t)n * sizeof(float);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    strided_copy<<<blocks, threads>>>(d_in, d_out, n, 1);  // 热身
    CUDA_CHECK_KERNEL();

    const int reps = 20;
    int strides[] = {1, 2, 4, 8, 16, 32};
    printf("%8s %12s %12s\n", "stride", "ms", "GB/s");
    for (int s : strides) {
        GpuTimer timer;
        timer.start();
        for (int r = 0; r < reps; r++)
            strided_copy<<<blocks, threads>>>(d_in, d_out, n, s);
        float ms = timer.stop_ms() / reps;
        CUDA_CHECK_KERNEL();
        // 读 + 写各 4 字节。
        double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;
        printf("%8d %12.4f %12.1f\n", s, ms, gbps);
    }
    return 0;
}
