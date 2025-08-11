'''
Author: mekeny1
Date: 2025-06-10 01:12:08
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:51:04
FilePath: \pycuda_tutorial_hapril\Chapter05\multi-kernel.py
Description: 使用PyCUDA演示多核函数串行执行的基础示例，展示GPU核函数的多次调用和基本的内存管理操作
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as dvr
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

"""
代码总体说明：
本程序演示了多核函数串行执行的基础实现，主要特点包括：

1. 算法思想：
   - 创建多个独立的数组，每个数组由GPU核函数独立处理
   - 通过串行调用GPU核函数实现多个数组的批量处理
   - 展示GPU核函数的基本调用模式和内存管理

2. GPU并行化策略：
   - 每个核函数调用处理一个独立的数组
   - 使用同步内存操作，确保操作的顺序性
   - 通过多次核函数调用实现批量数据处理

3. 内存管理特点：
   - 使用同步的GPU内存传输操作
   - 每个数组独立分配GPU内存空间
   - 通过同步获取确保计算结果的完整性

4. 软硬件特性利用：
   - GPU：串行执行多个核函数，每个处理独立的数据集
   - CPU：负责数据准备、核函数调用和结果收集
   - 内存：使用GPU全局内存，支持大规模数据处理
   - 同步操作：确保GPU操作的顺序性和数据一致性
"""

# 程序参数设置
num_arrays = 200  # 需要处理的数组数量
array_len = 1024**2  # 每个数组的长度（1M个元素）

# 定义CUDA核函数，执行数组的重复乘除运算
# 这个核函数主要用于演示GPU计算和内存操作
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

# 生成随机测试数据
for _ in range(num_arrays):
    # 创建随机浮点数数组，用于GPU计算测试
    data.append(np.random.randn(array_len).astype("float32"))

# 记录总体执行开始时间
t_start = time()

# 第一阶段：将数据传输到GPU内存
for k in range(num_arrays):
    # 将数据同步传输到GPU内存
    # 注意：这里使用同步传输，会阻塞CPU直到传输完成
    data_gpu.append(gpuarray.to_gpu(data[k]))

# 第二阶段：在GPU上执行核函数
for k in range(num_arrays):
    # 在GPU上执行核函数，处理对应的数组
    # block=(64,1,1)：每个块64个线程
    # grid=(1,1,1)：使用1个块
    # 注意：这里没有指定流，使用默认流，执行是串行的
    mult_ker(data_gpu[k], np.int32(array_len),
             block=(64, 1, 1), grid=(1, 1, 1))

# 第三阶段：从GPU获取计算结果
for k in range(num_arrays):
    # 从GPU同步获取计算结果
    # 注意：这里使用同步获取，会阻塞CPU直到传输完成
    gpu_out.append(data_gpu[k].get())

# 记录总体执行结束时间
t_end = time()

# 验证计算结果的正确性
for k in range(num_arrays):
    # 检查GPU计算结果是否与原始数据一致（考虑到浮点运算精度）
    assert (np.allclose(gpu_out[k], data[k]))

# 输出总体执行时间
print("Total time: {}".format(t_end-t_start))  # 注意：修正了原代码中的拼写错误"Toatal"
