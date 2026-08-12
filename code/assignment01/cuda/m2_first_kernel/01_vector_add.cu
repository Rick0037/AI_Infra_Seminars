// 问题 2.1：向量加法（填空）
// 六个空各考一个概念，填完编译运行，"PASS"即可。
// 填完之前这个文件无法通过编译。
#include "common.h"

// ====== 空 1：kernel 需要什么函数修饰符？ ======
/* 填这里 */ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // ====== 空 2：这个线程负责的全局下标 ======
    int idx = /* 填这里 */;
    // ====== 空 3：边界保护——总线程数可能多于元素个数 ======
    if (/* 填这里 */) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1000003;  // 故意取一个不是 256 整数倍的数
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ====== 空 4：把 h_a、h_b 拷到 device（注意最后一个方向参数） ======
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, /* 填这里 */));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, /* 填这里 */));

    int threadsPerBlock = 256;
    // ====== 空 5：block 数——向上取整，保证覆盖全部 n 个元素 ======
    int blocksPerGrid = /* 填这里 */;

    // ====== 空 6：启动 kernel（执行配置写在哪里？） ======
    vectorAdd /* 填这里 */ (d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}
