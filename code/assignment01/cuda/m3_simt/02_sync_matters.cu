// 问题 3.3：__syncthreads 实验。
// 每个 block 把自己的 256 个元素倒序：先搬进 shared memory，同步，
// 再交叉着读出来。任务：
//   1. 直接运行，确认 PASS；
//   2. 注释掉 __syncthreads() 那一行，再运行几次，观察结果；
//   3. 回答 handout 里的问题。
#include "common.h"

#define BLOCK 256

__global__ void reverse_blocks(const float *in, float *out, int n) {
    __shared__ float buf[BLOCK];
    int base = blockIdx.x * BLOCK;
    int t = threadIdx.x;

    buf[t] = in[base + t];
    __syncthreads();  // <-- 实验对象
    out[base + t] = buf[BLOCK - 1 - t];
}

int main() {
    const int nblocks = 4096;
    const int n = nblocks * BLOCK;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_in, n, 7);
    for (int b = 0; b < nblocks; b++)
        for (int t = 0; t < BLOCK; t++)
            h_ref[b * BLOCK + t] = h_in[b * BLOCK + (BLOCK - 1 - t)];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    reverse_blocks<<<nblocks, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_out, h_ref, n));
    return 0;
}
