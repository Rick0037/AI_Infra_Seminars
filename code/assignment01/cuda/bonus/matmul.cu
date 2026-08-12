// Bonus（选做）：naive 矩阵乘法。
// 任务：修改 BS（Block Size）的数值，每次改完后重新编译测试，统计 GFLOPS。
//   make bin/bonus/matmul && ./bin/bonus/matmul
//   nvcc -O2 -std=c++17 -I. -arch=native -DBS=8 -o bin/bonus/matmul_bs8 bonus/matmul.cu
#include "common.h"

#ifndef BS
#define BS 16
#endif

__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
    float *h_ref = (float *)malloc(bytesC);
    fill_random(h_A, (long)M * K, 1);
    fill_random(h_B, (long)K * N, 2);
    for (long i = 0; i < (long)M * K; i++) h_A[i] *= 0.1f;
    for (long i = 0; i < (long)K * N; i++) h_B[i] *= 0.1f;

    // CPU 参考（1024^3 次乘加，要几秒钟，耐心）。
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)h_A[i * K + k] * h_B[j + (size_t)k * N];
            h_ref[i * N + j] = (float)acc;
        }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 threads(BS, BS);
    dim3 blocks((N + BS - 1) / BS, (M + BS - 1) / BS);

    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);  // 热身
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    if (!check_close(h_C, h_ref, (long)M * N, 1e-2f)) REPORT(0);

    const int reps = 20;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < reps; r++)
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();

    double gflops = 2.0 * M * N * K / (ms * 1e-3) / 1e9;
    printf("BS=%d  平均 %.3f ms  %.1f GFLOPS\n", BS, ms, gflops);
    REPORT(1);
    return 0;
}
