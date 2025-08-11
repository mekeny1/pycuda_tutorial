'''
Author: mekeny1
Date: 2025-06-11 01:38:55
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:50:36
FilePath: \pycuda_tutorial_hapril\Chapter05\multi-kernel_streams.py
Description: 使用PyCUDA演示多核函数流(Streams)并发执行，通过多个CUDA流实现GPU计算的并行化，提高GPU利用率和整体计算性能
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

"""
代码总体说明：
本程序演示了多核函数流并发执行的实现，主要特点包括：

1. 算法思想：
   - 创建多个独立的CUDA流(Streams)，每个流处理一个独立的数组
   - 通过流的并发执行实现GPU计算的并行化
   - 利用GPU的硬件调度器实现真正的并行计算

2. GPU并行化策略：
   - 每个流独立执行核函数，避免流间的相互阻塞
   - 使用异步内存操作，减少CPU-GPU等待时间
   - 通过多流并发最大化GPU的并行处理能力

3. CUDA流机制：
   - 每个流维护独立的命令队列，支持并发执行
   - 异步操作与流绑定，实现精确的同步控制
   - 流的并发执行提高了GPU的整体吞吐量

4. 软硬件特性利用：
   - GPU：多流并发执行，充分利用GPU的并行计算能力
   - CPU：负责数据准备、流管理和结果验证
   - 内存：使用GPU全局内存，支持异步传输和并发访问
   - 流调度：GPU硬件调度器自动管理多个流的执行顺序
"""

# 程序参数设置
num_arrays = 200  # 并发处理的数组数量
array_len = 1024**2  # 每个数组的长度（1M个元素）

# 定义CUDA核函数，执行数组的重复乘除运算
# 这个核函数主要用于演示GPU计算和多流并发
ker = SourceModule(
    """
    __global__ void mult_ker(float *array, int arrayr_len)
    {
        int thd = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程的全局索引
        int num_iters = arrayr_len / blockDim.x;  // 计算每个线程需要处理的迭代次数

        // 每个线程处理多个数组元素，实现负载均衡
        for (int j = 0; j < num_iters; j++)
        {
            int i = j * blockDim.x + thd;  // 计算当前处理的数组索引

            // 执行50次乘除运算，增加计算负载用于性能测试
            for (int k = 0; k < 50; k++)
            {
                array[i] *= 2.0;  // 乘以2
                array[i] /= 2.0;  // 除以2（结果应该等于原值）
            }
        }
    }
"""
)

# 从编译后的模块中获取CUDA核函数
mult_ker = ker.get_function("mult_ker")

# 初始化数据结构
data = []  # 存储CPU端的原始数据
data_gpu = []  # 存储GPU端的数据
gpu_out = []  # 存储GPU计算结果
streams = []  # 存储CUDA流对象

# 创建多个CUDA流和生成测试数据
for _ in range(num_arrays):
    streams.append(drv.Stream())  # 为每个数组创建独立的CUDA流
    data.append(np.random.randn(array_len).astype("float32"))  # 生成随机测试数据

# 记录总体执行开始时间
t_start = time()

# 第一阶段：异步传输数据到GPU
for k in range(num_arrays):
    # 将数据异步传输到GPU，与对应的流关联
    # 每个流独立管理数据传输，支持并发执行
    data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))

# 第二阶段：在GPU上执行核函数
for k in range(num_arrays):
    # 在指定流上执行核函数
    # block=(64,1,1)：每个块64个线程
    # grid=(1,1,1)：使用1个块
    # stream=streams[k]：绑定到特定的CUDA流
    mult_ker(data_gpu[k], np.int32(array_len), block=(
        64, 1, 1), grid=(1, 1, 1), stream=streams[k])

# 第三阶段：异步获取计算结果
for k in range(num_arrays):
    # 从GPU异步获取计算结果，与对应的流关联
    # 每个流独立管理结果传输，支持并发执行
    gpu_out.append(data_gpu[k].get_async(stream=streams[k]))

# 记录总体执行结束时间
t_end = time()

# 验证计算结果的正确性
for k in range(num_arrays):
    # 检查GPU计算结果是否与原始数据一致（考虑到浮点运算精度）
    assert (np.allclose(gpu_out[k], data[k]))

# 输出总体执行时间
print("Total time: {}".format(t_end-t_start))  # 注意：修正了原代码中的拼写错误"Toatal"
