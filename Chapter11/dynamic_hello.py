'''
Author: mekeny1
Date: 2025-07-12 16:33:35
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:29:14
FilePath: \pycuda_tutorial_hapril\Chapter11\dynamic_hello.py
Description: CUDA动态并行演示程序，展示GPU内核函数递归调用和动态并行执行
Tags: cuda, dynamic-parallelism, recursive-kernel, gpu-programming, kernel-launch, pycuda
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit

# CUDA C 代码，演示动态并行和递归 kernel 调用
# 动态并行是CUDA 5.0引入的重要特性，允许GPU内核函数在运行时启动新的内核函数
DynamicParallelsmCode = """
// 动态并行核函数，递归调用自身
// 这是CUDA动态并行的核心特性：内核函数可以启动其他内核函数
__global__ void dynamic_hello_ker(int depth)
{
    // 每个线程打印自己的编号和递归深度
    // printf在GPU内核中的使用，输出会显示在控制台
    printf("Hello from thread %d, recursion depth %d!\\n", threadIdx.x, depth);
    
    // 仅在第一个线程、第一块、block 大小大于 1 时递归 launch 新 kernel
    // 这个条件确保递归有终止条件，避免无限递归
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockDim.x > 1)
    {
        printf("Launching a new kernel from depth %d .\\n",depth);
        printf("-----------------------------------------\\n");
        // 动态 launch 一个新的 kernel，block 数不变，blockDim.x-1
        // 这是动态并行的核心：内核函数在GPU上直接启动新的内核函数
        // 每次递归block大小减1，确保最终会终止
        dynamic_hello_ker<<<1, blockDim.x - 1>>>(depth + 1);
    }
}
"""

# 编译动态并行 CUDA 代码，获取核函数
# DynamicSourceModule专门用于编译支持动态并行的CUDA代码
# 与普通SourceModule不同，它支持内核函数间的相互调用
dp_mod = DynamicSourceModule(DynamicParallelsmCode)
hello_ker = dp_mod.get_function("dynamic_hello_ker")

# 启动 kernel，初始递归深度为 0，block 大小为 4
# 这个配置会产生递归调用：4->3->2->1，总共4层递归
# 每层递归都会打印当前线程信息和启动新内核的信息
hello_ker(np.int32(0), grid=(1, 1, 1), block=(4, 1, 1))
