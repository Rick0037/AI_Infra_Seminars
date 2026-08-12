import torch
import numpy as np


print("-----------Session 1.1-----------")
a = torch.tensor([1.0, 2.0, 3.0])
print(a)
print(a.shape)

zero = torch.zeros(3, 4)
print(zero)
print(zero.shape)
ones = torch.ones(2, 3, 4)
print(ones)
print(ones.shape)

print(ones[0])
print(ones[0].shape)

# fake random number generator
random_number = torch.randn(3, 4)

print(random_number)
print(random_number.shape)

# array range
# 0 start number
# 10 end number
# TODO :这里是有间隔的
seq = torch.arange(0, 10, 2)
print(seq)
print(seq.shape)

x = torch.randn(3, 4)
# same shape as x
y = torch.zeros_like(x)
z = torch.ones_like(y)

print(y)
print(z)

# shared memory copy as numpy
t = torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
print(t)

print("-----------Session 1.2-----------")
# TODO :这里是无间隔直接进行生成了
x = torch.arange(12)
print(x)

# if a.contiguous():
#     print("yes")
# else:
#     print(no)

# TODO view 要求 contiguous， reshape不要求
# TODO 一般都使用reshape， 除了array 之外很难说到底都是不是连续的内存
a = x.view(3, 4)
print(a)

# ---------出现报错了
# y = torch.randn(4, 4)
# a_y = x.view_as(y)
# print(a_y)
b = x.reshape(3, 4)

# TODO 在不知道维度的时候可以进行自动推导
c = x.view(3, -1)


# # permute 和 transpose 可以互相实现
# x.permute(1, 0, 2)  # 等价于 ↓
# x.transpose(0, 1)  # 交换 dim0 和 dim1

# (batch, seq_len, heads, dim)
# TODO 比较常见的都是 BSHD
tensor_test = torch.randn(2, 8, 12, 64)
print(tensor_test)
t_permuted = tensor_test.permute(2, 1, 0, 3)
# batch seq_size, mutli_head, dimsention in head
print("------")
print(t_permuted)
print(tensor_test.shape == t_permuted.shape)
print(t_permuted.shape)
t_permuted_transp = t_permuted.transpose(0, 2)
print(t_permuted_transp.shape)


e = torch.randn(1, 3, 1, 4)
f = e.squeeze()
print(e.squeeze().shape)
print(f.shape)
f = f.unsqueeze(0)
f = f.unsqueeze(1)
print(f.shape)


print("-----1.3----")
print(torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(3, 4)
    #  CPU → GPU（推荐写法）
    x_gpu = x.to("cuda")
    x_gpu = x.cuda()
    x_cpu = x_gpu.cpu()

    x_gps = torch.randn(3, 4, device="cuda")
    print(x_gps.device)
    print(x_cpu.device)
    print(x_cpu.get_device())


x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)  # 4 bytes/element
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # 2 bytes/element

# TODO nelement 一共有多少个element，
#  element_size 单个 element的size是多少，是多大
print(f"fp32: {x_fp32.nelement() * x_fp32.element_size() / 1024:.0f} KB")  # 3906 KB
print(f"bf16: {x_bf16.nelement() * x_bf16.element_size() / 1024:.0f} KB")  # 1953 KB

x = torch.randn(3, 4)  # 默认 fp32
x_half = x.half()  # 转 fp16
x_bf16 = x.to(torch.bfloat16)  # 转 bf16
