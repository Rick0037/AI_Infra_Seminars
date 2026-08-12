# 用法（在 cuda/ 目录下执行）：
#   make bin/m2_first_kernel/01_vector_add    编译单个练习
#   make run/m2_first_kernel/01_vector_add    编译并运行
#   ARCH=sm_80 make ...                       指定目标架构（默认 native，即当前这块卡）
#
# 模块 8 的编译实验（问题 8.2）：
#   make ptx/m0_env/01_hello             生成 PTX 文本，直接打开阅读
#   make sassonly/m0_env/01_hello        只嵌高架构 SASS（低架构卡上应当跑不起来）
#   make ptxonly/m0_env/01_hello         只嵌低架构 PTX（靠驱动 JIT 兜底）
#   ARCH_HIGH=100 make sassonly/...           如果你的卡本身就是 sm_90，把实验架构再调高

NVCC      ?= nvcc
ARCH      ?= native
ARCH_HIGH ?= 90
# CUDA 13 起 compute_70 已被移除，低架构实验默认用 75（CUDA 12/13 都支持）
ARCH_LOW  ?= 75
FLAGS     ?= -O2 -std=c++17 -I.

bin/%: %.cu common.h
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) -arch=$(ARCH) -o $@ $<

run/%: bin/%
	./$<

ptx/%: %.cu common.h
	$(NVCC) $(FLAGS) -ptx -o $*.ptx $<
	@echo "已生成 $*.ptx"

sassonly/%: %.cu common.h
	@mkdir -p $(dir bin/$*)
	$(NVCC) $(FLAGS) -gencode arch=compute_$(ARCH_HIGH),code=sm_$(ARCH_HIGH) -o bin/$*_sassonly $<
	@echo "已生成 bin/$*_sassonly（只含 sm_$(ARCH_HIGH) 的机器码）"

ptxonly/%: %.cu common.h
	@mkdir -p $(dir bin/$*)
	$(NVCC) $(FLAGS) -gencode arch=compute_$(ARCH_LOW),code=compute_$(ARCH_LOW) -o bin/$*_ptxonly $<
	@echo "已生成 bin/$*_ptxonly（只含 compute_$(ARCH_LOW) 的 PTX）"

clean:
	rm -rf bin */*.ptx

.PHONY: run/% ptx/% sassonly/% ptxonly/% clean
