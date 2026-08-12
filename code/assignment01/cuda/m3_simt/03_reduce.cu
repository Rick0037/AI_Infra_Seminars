// 问题 3.5：block 内求和归约。
//
// 任务：从零实现下面两个 kernel。main 里的判测与计时已经写好，不要改。
//
// contract（两个 kernel 相同的部分）：
//   - launch 配置是 <<<nblocks, BLOCK>>>，BLOCK = 256；
//   - 第 b 个 block 负责 in[b*BLOCK] 到 in[b*BLOCK + 255] 这 256 个元素，
//     把它们的和写进 out[b]（归约结束后由一个线程写出，通常是 tid == 0）；
//   - 求和在 shared memory 里做：每个线程先把自己对应的那个元素搬进
//     __shared__ float buf[BLOCK]，之后的加法全部在 buf 上进行；
//   - 每一轮配对加法前后都要 __syncthreads()
//
// 两个 kernel 的区别只在“每一轮由哪些线程做加法、加哪个位置”：
//   - reduce_interleaved（交错配对）：步长 s 取 1, 2, 4, ..., 128，
//     每轮由 tid % (2*s) == 0 的线程执行 buf[tid] += buf[tid + s]。
//     以 8 个元素为例：
//       s=1: buf[0]+=buf[1]  buf[2]+=buf[3]  buf[4]+=buf[5]  buf[6]+=buf[7]
//            （tid 0、2、4、6 干活，活跃线程在 warp 里隔一个一个）
//       s=2: buf[0]+=buf[2]  buf[4]+=buf[6]   （tid 0、4 干活）
//       s=4: buf[0]+=buf[4]                   （tid 0 干活）
//   - reduce_contiguous（连续配对）：步长 s 取 128, 64, ..., 1，
//     每轮由 tid < s 的线程执行 buf[tid] += buf[tid + s]。同样 8 个元素：
//       s=4: buf[0]+=buf[4]  buf[1]+=buf[5]  buf[2]+=buf[6]  buf[3]+=buf[7]
//            （tid 0-3 干活，活跃线程挤在编号低的一头）
//       s=2: buf[0]+=buf[2]  buf[1]+=buf[3]   （tid 0、1 干活）
//       s=1: buf[0]+=buf[1]                   （tid 0 干活）
//   两版加法次数完全相同，区别只在活跃线程在 warp 里怎么分布
//
// 注意：两个 kernel 里循环的上下界都用 blockDim.x（运行期值）来写，
// 不要用宏 BLOCK。用编译期常量会让编译器把循环完全展开、把 % 优化成
// 位运算，影响性能比较。
//
// 写完运行，两个版本都 PASS 后，记录耗时和比值并解释差距。
//
// 选做第三版（shuffle 版）时：测试只会跑上面两个 kernel，自己在 main 里
// 照着加一次 run_one 调用即可，不影响前两版的测试。
#include "common.h"

#define BLOCK 256

__global__ void reduce_interleaved(const float *in, float *out) {
    // TODO：从这里开始写（交错配对版本）
}

__global__ void reduce_contiguous(const float *in, float *out) {
    // TODO：从这里开始写（连续配对版本）
}

// ---------------- 以下是判测与计时，不要修改 ----------------

typedef void (*reduce_fn)(const float *, float *);

static float run_one(reduce_fn fn, const char *name, const float *d_in,
                     float *d_out, float *h_out, const float *h_partial,
                     int nblocks) {
    CUDA_CHECK(cudaMemset(d_out, 0, nblocks * sizeof(float)));
    fn<<<nblocks, BLOCK>>>(d_in, d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nblocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (!check_close(h_out, h_partial, nblocks, 1e-3f)) {
        printf("%s: FAIL\n", name);
        emit_result("3.5", "fail", "{}");
        exit(1);
    }

    const int reps = 200;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < reps; r++) fn<<<nblocks, BLOCK>>>(d_in, d_out);
    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();
    printf("%s: PASS  平均 %.4f ms\n", name, ms);
    return ms;
}

int main() {
    const int nblocks = 4096;
    const int n = nblocks * BLOCK;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(nblocks * sizeof(float));
    float *h_partial = (float *)malloc(nblocks * sizeof(float));
    fill_random(h_in, n, 11);
    for (int b = 0; b < nblocks; b++) {
        double s = 0;
        for (int t = 0; t < BLOCK; t++) s += h_in[b * BLOCK + t];
        h_partial[b] = (float)s;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, nblocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    float ms_i = run_one(reduce_interleaved, "interleaved", d_in, d_out, h_out,
                         h_partial, nblocks);
    float ms_c = run_one(reduce_contiguous, "contiguous ", d_in, d_out, h_out,
                         h_partial, nblocks);
    // 阈值 1.5x：A100 实测 2.22x、V100 实测 2.33x，两版写成一样时是 ~1x。
    float ratio = report_speedup("interleaved / contiguous", ms_i, ms_c, 1.5f,
                                 "两版耗时几乎一样，检查是不是写成同一个实现了");

    char metrics[192];
    snprintf(metrics, sizeof(metrics),
             "{\"interleaved_ms\":%.4f,\"contiguous_ms\":%.4f,\"ratio\":%.3f}",
             ms_i, ms_c, ratio);
    emit_result("3.5", "pass", metrics);
    return 0;
}
