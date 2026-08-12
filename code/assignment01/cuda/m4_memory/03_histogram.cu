// 问题 4.5：直方图（填空）。
// 统计 16M 个字节的值落在 256 个 bucket 里的次数。
// 注意：多个线程可能同时修改同一个 bucket 的值。
#include "common.h"

__global__ void histogram(const unsigned char *data, unsigned int *hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        unsigned char v = data[i];
        // ====== 空 1：往 hist[v] 里加 1
        //         该用哪个原子操作？ ======
        /* 填这里 */;
    }
}

int main() {
    const int n = 1 << 24;
    const int BINS = 256;

    unsigned char *h_data = (unsigned char *)malloc(n);
    unsigned int h_hist[BINS], h_ref[BINS] = {0};
    srand(9);
    for (int i = 0; i < n; i++) h_data[i] = (unsigned char)(rand() % BINS);
    for (int i = 0; i < n; i++) h_ref[h_data[i]]++;

    unsigned char *d_data;
    unsigned int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));

    int threads = 256, blocks = 1024;
    histogram<<<blocks, threads>>>(d_data, d_hist, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    int ok = 1;
    for (int b = 0; b < BINS; b++)
        if (h_hist[b] != h_ref[b]) {
            fprintf(stderr, "bin %d: got %u, want %u\n", b, h_hist[b], h_ref[b]);
            ok = 0;
            break;
        }

    // 计时，供问题 4.6 改造后对比。
    const int reps = 50;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < reps; r++)
        histogram<<<blocks, threads>>>(d_data, d_hist, n);
    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();
    printf("平均耗时 %.4f ms  (%.2f GB/s)\n", ms, n / ms / 1e6);
    REPORT(ok);
    return 0;
}
