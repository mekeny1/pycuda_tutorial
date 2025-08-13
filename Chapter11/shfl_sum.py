'''
Author: mekeny1
Date: 2025-07-15 20:10:30
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:29:37
FilePath: \pycuda_tutorial_hapril\Chapter11\shfl_sum.py
Description: CUDA warp shuffle指令求和演示，展示线程束内高效归约算法和同步通信机制
Tags: cuda, warp-shuffle, thread-reduction, gpu-computing, parallel-algorithms, synchronization, shfl-down-sync
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# CUDA C 代码，演示使用 __shfl_down_sync 进行线程束内归约求和
# warp shuffle是GPU上最高效的线程间通信方式，避免了共享内存的bank冲突
ShflSumCode = """
__global__ void shfl_sum_ker(int *input, int *output)
{
    int temp = input[threadIdx.x]; // 每个线程读取一个输入元素
    // 每个线程负责处理一个数组元素，这是并行归约的基础

    // 线程束内归约求和，利用 shuffle 指令高效通信
    // 这是GPU上最高效的归约算法，时间复杂度为O(log n)
    for (int i = 1; i < 32; i *= 2)
    {
        // mask 控制哪些线程参与 shuffle 操作。一般用 0xFFFFFFFF 表示全部线程参与，特殊情况下可以用自定义的 mask 或 __activemask() 以保证正确性和安全性。
        // __shfl_down_sync是Volta架构引入的同步shuffle指令，确保线程间的正确同步
        // 参数说明：mask(0xFFFFFFFF), value(temp), offset(i), width(32)
        // i表示向下偏移的距离，每次迭代翻倍，实现对数级归约
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i, 32);
    }

    // 线程束内第一个线程写出最终结果
    // 经过log2(32)=5次迭代后，线程0包含了所有32个元素的和
    if (threadIdx.x == 0)
    {
        *output = temp;
    }
}
"""

# 编译 CUDA 代码，获取核函数
# 这个核函数使用了现代GPU架构的shuffle指令，需要Volta或更新架构支持
shfl_mod = SourceModule(ShflSumCode)
shfl_sum_ker = shfl_mod.get_function("shfl_sum_ker")

# 构造输入数据（0~31）并拷贝到 GPU
# 使用32个连续整数作为测试数据，便于验证归约结果的正确性
array_in = gpuarray.to_gpu(np.int32(range(32)))
output = gpuarray.empty((1,), dtype=np.int32)

# 启动核函数，block 大小为 32
# block大小为32确保每个block正好是一个warp，最大化shuffle指令效率
shfl_sum_ker(array_in, output, grid=(1, 1, 1), block=(32, 1, 1))

print("Input array: %s" % array_in.get())
print("Summed value: %s" % output.get()[0])
# 校验 GPU 归约结果与 Python sum 是否一致
# 验证GPU并行归约算法的正确性，确保结果与串行计算一致
print("Does this match with python's sum? : %s" %
      (output.get()[0] == sum(array_in.get())))
