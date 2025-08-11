'''
Author: mekeny1
Date: 2025-05-29 00:56:34
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:40:09
FilePath: \pycuda_tutorial_hapril\Chapter04\simple_scalar_multiply_kernel.py
Description: 使用PyCUDA实现标量乘法内核，展示GPU并行计算在向量运算中的基础应用，每个线程独立处理一个数组元素
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as dvr
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

# =============================================================================
# 标量乘法GPU内核实现 - 基础并行计算示例
# =============================================================================
# 算法核心思想：
# 1. 数据并行：每个线程处理一个数组元素，实现完全并行化
# 2. 内存合并访问：相邻线程访问相邻内存地址，最大化内存带宽
# 3. 无数据依赖：每个线程独立计算，无需同步机制
#
# 硬件特性利用：
# - GPU并行计算：512个线程同时执行，理论加速比512倍
# - 内存带宽：所有线程同时访问内存，充分利用内存带宽
# - SIMT架构：单指令多线程，提高指令执行效率
# =============================================================================

ker = SourceModule(
    """
    // 标量乘法内核函数：每个线程将向量元素与标量相乘
    __global__ void scalar_multiply_kernel(float *outVec, float scalar, float *vec)
    {
        int i=threadIdx.x;  // 获取线程ID作为数组索引
        outVec[i]=scalar*vec[i];  // 执行标量乘法并存储结果
    }
    """
)


scalar_multiply_gpu = ker.get_function("scalar_multiply_kernel")

# 创建测试数据：512个随机单精度浮点数
testVec = np.random.randn(512).astype(np.float32)
testVec_gpu = gpuarray.to_gpu(testVec)
outVec_gpu = gpuarray.empty_like(testVec_gpu)

# 执行GPU内核：512个线程并行计算，每个线程处理一个元素
scalar_multiply_gpu(outVec_gpu, np.float32(2), testVec_gpu,
                    block=(512, 1, 1), grid=(1, 1, 1,))

# 验证结果：GPU计算结果与CPU计算结果比较
print("Does our kernel work correctly? : {}".format(
    np.allclose(outVec_gpu.get(), 2*testVec)))
