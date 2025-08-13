'''
Author: mekeny1
Date: 2025-07-13 22:58:24
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:29:58
FilePath: \pycuda_tutorial_hapril\Chapter11\vectorized_memory.py
Description: CUDA向量化内存访问演示，展示int4和double2向量类型的内存带宽优化技术
Tags: cuda, vectorized-memory, memory-bandwidth, gpu-optimization, int4-double2, memory-access, gpu-computing
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# CUDA C 代码，演示向量化内存访问（int4, double2）
# 向量化内存访问是GPU性能优化的重要技术，通过一次内存事务读取多个数据元素
VecCode = """
__global__ void vec_ker(int *ints, double *doubles)
{
    int4 f1, f2;
    // int4是CUDA的向量类型，包含4个32位整数，可以一次性读取16字节数据

    // 通过 int4 类型一次性读取 4 个 int
    // reinterpret_cast将指针转换为int4*类型，实现向量化内存访问
    // 这种访问方式比逐个读取4个int更高效，减少内存事务数量
    f1 = *reinterpret_cast<int4 *>(ints);
    f2 = *reinterpret_cast<int4 *>(&ints[4]);

    printf("First int4: %d, %d, %d, %d\\n", f1.x, f1.y, f1.z, f1.w);
    printf("Second int4: %d, %d, %d, %d\\n", f2.x, f2.y, f2.z, f2.w);

    double2 d1, d2;
    // double2是CUDA的向量类型，包含2个64位双精度浮点数，可以一次性读取16字节数据

    // 通过 double2 类型一次性读取 2 个 double
    // 同样使用reinterpret_cast实现向量化访问，提高内存带宽利用率
    d1 = *reinterpret_cast<double2 *>(doubles);
    d2 = *reinterpret_cast<double2 *>(&doubles[2]);

    printf("First double2: %f, %f\\n", d1.x, d1.y);
    printf("Second double2: %f, %f\\n", d2.x, d2.y);
}
"""

# 编译 CUDA 代码，获取核函数
# 向量化内存访问需要正确的内存对齐，编译器会优化内存访问模式
vec_mod = SourceModule(VecCode)
vec_ker = vec_mod.get_function("vec_ker")

# 构造输入数据，8 个 int 和 4 个 double，并拷贝到 GPU
# 数据量选择考虑了向量化访问的边界，确保内存对齐
ints = gpuarray.to_gpu(np.int32([1, 2, 3, 4, 5, 6, 7, 8]))
doubles = gpuarray.to_gpu(np.double([1.11, 2.22, 3.33, 4.44]))

print("Vectorized Memory Test:")
# 启动核函数，测试向量化内存访问
# 使用单线程执行，便于观察向量化内存访问的效果
vec_ker(ints, doubles, grid=(1, 1, 1), block=(1, 1, 1))
