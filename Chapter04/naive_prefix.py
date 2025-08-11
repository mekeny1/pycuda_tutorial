'''
Author: mekeny1
Date: 2025-06-02 22:10:53
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:38:43
FilePath: \pycuda_tutorial_hapril\Chapter04\naive_prefix.py
Description: 使用PyCUDA实现并行前缀和算法，采用共享内存和递归倍增技术，展示GPU并行计算在数组累加运算中的高效应用
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as dvr
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time

# =============================================================================
# 并行前缀和算法GPU实现 - 递归倍增版本
# =============================================================================
# 算法核心思想：
# 1. 递归倍增策略：每次迭代将步长翻倍，实现O(log n)的时间复杂度
# 2. 共享内存优化：使用__shared__内存减少全局内存访问延迟
# 3. 线程同步：通过__syncthreads()确保数据依赖关系正确
#
# 硬件特性利用：
# - GPU共享内存：1024个线程共享快速内存，访问延迟极低
# - SIMT架构：所有线程并行执行相同指令，提高计算效率
# - 内存带宽：1024个线程同时访问内存，最大化内存带宽利用率
# =============================================================================

naive_ker = SourceModule(
    """
    __global__ void naive_prefix(double *vec, double *out)
    {
        __shared__ double sum_buf[1024];  // 声明1024个双精度数的共享内存数组
        int tid = threadIdx.x;  // 获取线程ID
        sum_buf[tid] = vec[tid];  // 将全局内存数据复制到共享内存

        // 并行前缀和算法核心：递归倍增策略
        // 算法原理：每次迭代将步长翻倍，实现对数时间复杂度的累加
        // 以tid=1023为例：
        // iter=1时：加上tid=1022的值
        // iter=2时：再加上tid=1021的值（此时tid=1021已包含tid=1020的值）
        // iter=4时：再加上tid=1019的值（此时tid=1019已包含tid=1018,1017,1016的值）
        // 以此类推，最终tid=1023将包含所有前面元素的和
        int iter = 1;
        for (int i = 0; i < 10; i++)  // 10次迭代：log2(1024) = 10
        {
            __syncthreads();  // 同步点：确保所有线程完成当前迭代
            if (tid >= iter)  // 只有索引大于等于步长的线程参与计算
            {
                sum_buf[tid] = sum_buf[tid] + sum_buf[tid - iter];  // 递归累加
            }
            iter *= 2;  // 步长翻倍：1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        }
        __syncthreads();  // 最终同步：确保所有累加操作完成
        out[tid]=sum_buf[tid];  // 将结果写回全局内存
        __syncthreads();  // 输出同步：确保所有写操作完成
    }
    """
)


naive_gpu = naive_ker.get_function("naive_prefix")


if __name__ == "__main__":
    # 创建测试向量：1024个随机双精度数
    testvec = np.random.randn(1024).astype(np.float64)
    testvec_gpu = gpuarray.to_gpu(testvec)
    outvec_gpu = gpuarray.empty_like(testvec_gpu)

    # 执行GPU内核：1个Grid，1024个线程的Block
    # 1024个线程并行计算前缀和，时间复杂度O(log n)
    naive_gpu(testvec_gpu, outvec_gpu, block=(1024, 1, 1), grid=(1, 1, 1))

    # 验证结果：CPU计算总和与GPU最后一个元素比较
    total_sum = sum(testvec)
    total_sum_gpu = outvec_gpu[-1].get()

    print("Does our kernel work correctly? : {}".format(
        np.allclose(total_sum_gpu, total_sum)))
