// 问题 4.8：occupancy 实验。
// 思路：shared memory 按 block 分配，一个 block 占得越多，SM 上能同时
// 驻留的 block 就越少，常驻 warp 数（occupancy）随之下降。下面的 kernel
// 声明了不实际用于储值的动态 shared memory——计算量和访存量完全不变，
// 变的只有 SM 上的并行度。
// 程序对每一档 shared memory 用量做两件事：
//   1. 用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 查询这一档下
//      每个 SM 理论上能驻留几个 block，换算成 occupancy；
//   2. 实测同一个逐元素加法 kernel 的有效带宽。
// 记录实测数据，并回答相关问题
#include "common.h"

#define BLOCK 256

__global__ void stream_add(const float *a, const float *b, float *c, int n) {
    extern __shared__ float ballast[];  // 只占 shared memory，不使用
    (void)ballast;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int smem_sm = (int)prop.sharedMemPerMultiprocessor;
    int smem_blk_max = (int)prop.sharedMemPerBlockOptin;
    int max_threads = prop.maxThreadsPerMultiProcessor;
    printf("%s：shared memory %d KB / SM，最大常驻 %d 线程 / SM\n\n",
           prop.name, smem_sm / 1024, max_threads);

    // 允许单个 block 申请超过默认上限（48 KB）的动态 shared memory
    CUDA_CHECK(cudaFuncSetAttribute((const void *)stream_add,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_blk_max));

    const int n = 1 << 26;
    size_t bytes = (size_t)n * sizeof(float);
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));
    int nblocks = (n + BLOCK - 1) / BLOCK;

    // shared memory 档位：按每 SM 总量的比例给定。一个 block 占了总量的
    // 1/x，每 SM 大致就只能驻留 x 个 block。下面六档挑得能在多数卡上落到
    // 六个不同的驻留块数，但实际落点还受架构影响（有的架构给每个 block
    // 额外保留一小块 shared），一切以 API 报出来的数为准。
    const double fracs[] = {0.0, 0.132, 0.15, 0.18, 0.29, 0.55};
    printf("%-14s %-16s %-11s %s\n",
           "shared/block", "理论 block/SM", "occupancy", "实测带宽");
    for (int k = 0; k < 6; k++) {
        int smem = (int)(smem_sm * fracs[k]);
        if (smem > smem_blk_max) smem = smem_blk_max;

        int active = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, stream_add, BLOCK, smem));
        double occ = 100.0 * active * BLOCK / max_threads;

        stream_add<<<nblocks, BLOCK, smem>>>(d_a, d_b, d_c, n);  // 热身
        CUDA_CHECK_KERNEL();
        const int reps = 20;
        GpuTimer timer;
        timer.start();
        for (int r = 0; r < reps; r++)
            stream_add<<<nblocks, BLOCK, smem>>>(d_a, d_b, d_c, n);
        float ms = timer.stop_ms() / reps;
        CUDA_CHECK_KERNEL();
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;

        printf("%8.1f KB %10d %14.1f%% %10.1f GB/s\n",
               smem / 1024.0, active, occ, gbps);
    }

    // 讲义里提到的另一个 API：让 runtime 建议一个 occupancy 最高的 block size
    int min_grid = 0, best_block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &best_block, stream_add, 0, 0));
    printf("\ncudaOccupancyMaxPotentialBlockSize 建议（smem = 0 时）：blockSize = %d\n",
           best_block);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
