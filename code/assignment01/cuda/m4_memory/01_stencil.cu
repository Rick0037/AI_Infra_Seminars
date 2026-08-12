// 问题 4.2：三点平均 stencil（填空）。
// out[i] = (in[i-1] + in[i] + in[i+1]) / 3，越界位置按 0 处理。
// 写两个 kernel：一个用静态 shared memory，一个用动态的。
// 填完之前这个文件无法通过编译。
// 注：设置block内共享tile数组是为了减少对global memory的访问次数——
//    每个元素只从显存读一次，块内的三次复用都走片上的 shared memory
//    （延迟远低于显存）。
#include "common.h"

#define BLOCK 256
#define RADIUS 1

__global__ void stencil_static(const float *in, float *out, int n) {
    // ====== 空 1：静态 shared 数组，要装下 BLOCK 个元素加两侧 halo ======
    __shared__ float tile[/* 填这里 */];

    int g = blockIdx.x * blockDim.x + threadIdx.x;  // 全局下标
    int l = threadIdx.x + RADIUS;                   // 在 tile 里的位置

    tile[l] = (g < n) ? in[g] : 0.f;
    // 块两端的线程多搬一个 halo 元素。
    if (threadIdx.x < RADIUS) {
        int left = g - RADIUS;
        int right = g + BLOCK;
        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }

    // ====== 空 2：在这里补上一行 ======
    /* 填这里 */

    if (g < n) {
        // ====== 空 3：用 tile（不许用 in）算三点平均 ======
        out[g] = /* 填这里 */;
    }
}

__global__ void stencil_dynamic(const float *in, float *out, int n) {
    // ====== 空 4：动态 shared 数组的声明方式（大小在 launch 时才给出） ======
    /* 填这里（声明动态 shared 数组 tile）*/

    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int l = threadIdx.x + RADIUS;

    tile[l] = (g < n) ? in[g] : 0.f;
    if (threadIdx.x < RADIUS) {
        int left = g - RADIUS;
        int right = g + BLOCK;
        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }
    __syncthreads();
    if (g < n) {
        out[g] = (tile[l - 1] + tile[l] + tile[l + 1]) / 3.f;
    }
}

int main() {
    const int n = 1000003;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_in, n, 3);
    for (int i = 0; i < n; i++) {
        float l = (i > 0) ? h_in[i - 1] : 0.f;
        float r = (i < n - 1) ? h_in[i + 1] : 0.f;
        h_ref[i] = (l + h_in[i] + r) / 3.f;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK - 1) / BLOCK;

    stencil_static<<<blocks, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_out, h_ref, n)) REPORT(0);
    printf("static  PASS\n");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    // ====== 空 5：动态 shared 版本的 launch——第三个参数该填多少字节？ ======
    stencil_dynamic<<<blocks, BLOCK, /* 填这里 */>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_out, h_ref, n)) REPORT(0);
    printf("dynamic PASS\n");

    REPORT(1);
    return 0;
}
