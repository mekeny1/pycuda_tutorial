'''
Author: mekeny1
Date: 2025-07-15 11:08:50
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:29:12
FilePath: \pycuda_tutorial_hapril\Chapter11\atomic.py
Description: CUDA原子操作演示程序，展示GPU并行计算中的线程安全操作
Tags: cuda, atomic-operations, gpu-computing, parallel-programming, thread-safety, pycuda
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv

# CUDA C 代码，定义了演示原子操作的核函数
# 原子操作是GPU并行计算中的关键概念，确保多线程环境下的数据一致性
AtomicCode = """
// 核函数，演示原子加法、原子最大值、原子交换
// 原子操作确保在多个线程同时访问同一内存位置时的数据完整性
__global__ void atomic_ker(int *add_out,int *max_out)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x; // 计算全局线程索引
    // 这是CUDA中计算线程ID的标准公式，确保每个线程有唯一的标识符

    // 原子交换：将 *add_out 设置为 0，所有线程安全地执行
    // atomicExch是CUDA提供的原子操作，保证操作的原子性（不可分割性）
    // 即使多个线程同时执行，也只有一个线程能成功将值设为0
    atomicExch(add_out,0);
    __syncthreads(); // 线程同步，确保所有线程都已完成上一步
    // __syncthreads()是CUDA的同步原语，确保同一block内的所有线程都执行到此点

    // 原子加法：每个线程对 *add_out 加 1
    // atomicAdd确保即使多个线程同时执行加法，结果也是正确的累加值
    // 这是并行归约（reduction）操作的基础
    atomicAdd(add_out,1);

    // 原子最大值：所有线程将自己的 tid 与 max_out 比较，写入最大值
    // atomicMax用于并行查找最大值，常用于并行算法中的极值计算
    atomicMax(max_out,tid);
}
"""

# 编译 CUDA 代码，获取核函数
# SourceModule将CUDA C代码编译为GPU可执行的PTX代码
atomic_mod = SourceModule(AtomicCode)
atomic_ker = atomic_mod.get_function("atomic_ker")

# 分配 GPU 输出内存
# gpuarray.empty创建GPU内存空间，用于存储计算结果
# dtype=np.int32确保数据类型与CUDA代码中的int类型匹配
add_out = gpuarray.empty((1,), dtype=np.int32)
max_out = gpuarray.empty((1,), dtype=np.int32)

# 启动核函数，block 里有 100 个线程
# grid=(1,1,1)表示使用1个block，block=(100,1,1)表示每个block有100个线程
# 这种配置适合演示原子操作，因为所有线程都会竞争访问同一内存位置
atomic_ker(add_out, max_out, grid=(1, 1, 1), block=(100, 1, 1))
# 同步，确保 GPU 计算完成
# Context.synchronize()确保CPU等待GPU完成所有计算任务
drv.Context.synchronize()

print("Atomic operations test: ")
# 打印原子加法结果（应为线程数 100）
# 由于所有100个线程都执行了atomicAdd(add_out,1)，最终结果应该是100
print("add_out: %s" % add_out.get()[0])
# 打印原子最大值结果（应为最大线程 id 99）
# 线程ID从0开始，所以100个线程的ID范围是0-99，最大值应该是99
print("max_out: %s" % max_out.get()[0])
