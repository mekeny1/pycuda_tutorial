'''
Author: mekeny1
Date: 2025-06-04 20:05:07
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:44:01
FilePath: \pycuda_tutorial_hapril\Chapter04\work-efficient_prefix.py
Description: 使用PyCUDA实现工作高效的前缀和算法，采用上下扫描策略和动态线程块配置，展示GPU并行计算在大型数组累加运算中的优化应用
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

# =============================================================================
# 工作高效前缀和算法GPU实现 - 上下扫描策略版本
# =============================================================================
# 算法核心思想：
# 1. 上下扫描策略：先向上扫描构建部分和，再向下扫描计算最终前缀和
# 2. 动态线程块配置：根据数据规模动态调整线程块大小，提高资源利用率
# 3. 工作高效：总工作量为O(n)，优于naive版本的O(n log n)
#
# 硬件特性利用：
# - 动态线程管理：根据计算需求调整线程块大小，避免资源浪费
# - 内存访问优化：使用双缓冲技术避免读写冲突
# - 并行计算：支持大规模数组(32K元素)的高效处理
# =============================================================================

# 向上扫描内核：构建部分和数组
up_ker = SourceModule(
    """
    __global__ void up_ker(double *x, double *x_old, int k)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 计算全局线程ID

        int _2k = 1 << k;      // 2^k
        int _2k1 = 1 << (k + 1);  // 2^(k+1)

        int j = tid * _2k1;    // 计算当前线程负责的起始位置

        // 执行向上扫描：将两个相邻区间合并
        x[j + _2k1 - 1] = x_old[j + _2k - 1] + x_old[j + _2k1 - 1];
    }
    """
)
up_gpu = up_ker.get_function("up_ker")


def up_sweep(x):
    """向上扫描函数：构建部分和数组"""
    x = np.float64(x)
    x_gpu = gpuarray.to_gpu(np.float64(x))
    x_old_gpu = x_gpu.copy()  # 创建输入缓冲区副本

    # 执行log2(n)次向上扫描
    for k in range(int(np.log2(x.size))):
        num_threads = int(np.ceil(x.size/2**(k+1)))  # 计算当前迭代需要的线程数
        grid_size = int(np.ceil(num_threads/32))     # 计算网格大小

        # 动态调整线程块大小：优先使用32线程块，不足时使用剩余线程数
        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads

        # 执行GPU内核：动态配置线程块和网格
        up_gpu(x_gpu, x_old_gpu, np.int32(k), block=(
            block_size, 1, 1), grid=(grid_size, 1, 1))
        x_old_gpu[:] = x_gpu[:]  # 更新输入缓冲区

    x_out = x_gpu.get()
    return x_out


# 向下扫描内核：计算最终前缀和
down_ker = SourceModule(
    """
    __global__ void down_ker(double *y, double *y_old, int k)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 计算全局线程ID

        int _2k = 1 << k;      // 2^k
        int _2k1 = 1 << (k + 1);  // 2^(k+1)

        int j = tid * _2k1;    // 计算当前线程负责的起始位置

        // 执行向下扫描：计算最终前缀和
        y[j + _2k - 1] = y_old[j + _2k1 - 1];  // 保持左半部分不变
        y[j + _2k1 - 1] = y_old[j + _2k1 - 1] + y_old[j + _2k - 1];  // 更新右半部分
    }
    """
)
down_gpu = down_ker.get_function("down_ker")


def down_sweep(y):
    """向下扫描函数：计算最终前缀和"""
    y = np.float64(y)
    y[-1] = 0  # 设置最后一个元素为0
    y_gpu = gpuarray.to_gpu(y)
    y_old_gpu = y_gpu.copy()  # 创建输入缓冲区副本

    # 反向执行log2(n)次向下扫描
    for k in reversed(range(int(np.log2(y.size)))):
        num_threads = int(np.ceil(y.size/2**(k+1)))  # 计算当前迭代需要的线程数
        grid_size = int(np.ceil(num_threads/32))     # 计算网格大小

        # 动态调整线程块大小
        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads

        # 执行GPU内核：动态配置线程块和网格
        down_gpu(y_gpu, y_old_gpu, np.int32(k), block=(
            block_size, 1, 1), grid=(grid_size, 1, 1))
        y_old_gpu[:] = y_gpu[:]  # 更新输入缓冲区

    y_out = y_gpu.get()
    return y_out


def efficient_prefix(x):
    """工作高效前缀和主函数：先向上扫描，再向下扫描"""
    return down_sweep(up_sweep(x))


if __name__ == "__main__":
    # 创建测试向量：32K个随机双精度数
    testvec = np.random.randn(32*1024).astype(np.float64)
    testvec_gpu = gpuarray.to_gpu(testvec)

    outvec_gpu = gpuarray.empty_like(testvec_gpu)

    # CPU计算前缀和作为参考结果
    prefix_sum = np.roll(np.cumsum(testvec), 1)
    prefix_sum[0] = 0

    # GPU计算前缀和
    prefix_sum_gpu = efficient_prefix(testvec)

    # 验证结果正确性
    print("Does our work-efficient prefix work? {}".format(np.allclose(prefix_sum_gpu, prefix_sum)))
