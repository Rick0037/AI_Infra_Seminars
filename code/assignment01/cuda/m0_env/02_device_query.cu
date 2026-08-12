// 问题 0.2：查询你手上这块卡（填空）。
// 空的内容都是 cudaDeviceProp 的字段名，查 CUDA Runtime API 文档
// （搜 "cudaDeviceProp"），把五个空补上。
// 填完之前这个文件无法编译。
#include "common.h"

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU 型号            : %s\n", prop.name);
    printf("compute capability  : %d.%d\n", prop.major, prop.minor);

    // ====== 空 1：SM 数量（提示：字段名以 multiProcessor 开头） ======
    printf("SM 数量             : %d\n", /* 填这里 */);

    // ====== 空 2：warp 大小 ======
    printf("warp 大小           : %d\n", /* 填这里 */);

    // ====== 空 3：每个 block 可用的 shared memory 上限（字节） ======
    printf("shared mem / block  : %zu\n", (size_t) /* 填这里 */);

    // ====== 空 4：每个 SM 的最大常驻线程数 ======
    printf("max threads / SM    : %d\n", /* 填这里 */);

    // ====== 空 5：全局显存总量（字节） ======
    printf("global mem          : %zu\n", (size_t) /* 填这里 */);

    printf("max threads / block : %d\n", prop.maxThreadsPerBlock);
    return 0;
}
