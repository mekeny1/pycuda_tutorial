'''
Author: mekeny1
Date: 2025-06-13 00:32:07
LastEditors: mekeny1
LastEditTime: 2025-08-11 11:25:37
FilePath: \pycuda_tutorial_hapril\Chapter05\multi-kernel_events.py
Description: 使用PyCUDA演示多核函数并发执行和CUDA事件(Events)机制，实现精确的GPU内核性能测量和异步操作管理
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
本程序演示了多核函数并发执行和CUDA事件机制的使用，主要特点包括：

1. 算法思想：
   - 创建多个独立的GPU核函数实例，每个处理一个大型数组
   - 使用CUDA事件(Events)精确测量每个核函数的执行时间
   - 通过多流并发执行，最大化GPU利用率

2. GPU并行化策略：
   - 每个核函数处理一个独立的数组，实现真正的并行计算
   - 使用多个CUDA流(Streams)实现并发执行
   - 通过事件机制实现精确的时间测量和同步控制

3. CUDA事件机制：
   - 使用start_events和end_events记录每个核函数的开始和结束时间
   - 通过time_till()方法计算精确的内核执行时间
   - 事件与特定流关联，确保测量的准确性

4. 软硬件特性利用：
   - GPU：多核函数并发执行，充分利用GPU的并行计算能力
   - CPU：负责数据准备、事件管理和结果验证
   - 内存：使用GPU全局内存存储大型数组，支持异步传输
   - 流和事件：实现精确的性能测量和异步操作管理
"""

# 定义CUDA核函数，执行数组的重复乘除运算
# 这个核函数主要用于演示GPU计算和性能测量
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

# 程序参数设置
num_arrays = 200  # 并发处理的数组数量
array_len = 1024**2  # 每个数组的长度（1M个元素）

# 初始化数据结构
data = []  # 存储CPU端的原始数据
data_gpu = []  # 存储GPU端的数据
gpu_out = []  # 存储GPU计算结果
streams = []  # 存储CUDA流对象
start_events = []  # 存储开始事件对象
end_events = []  # 存储结束事件对象

# 为每个数组创建独立的CUDA流和事件
for _ in range(num_arrays):
    streams.append(drv.Stream())  # 创建新的CUDA流
    start_events.append(drv.Event())  # 创建开始事件
    end_events.append(drv.Event())  # 创建结束事件

# 生成随机测试数据
for _ in range(num_arrays):
    # 创建随机浮点数数组，用于GPU计算测试
    data.append(np.random.randn(array_len).astype("float32"))

# 记录总体执行开始时间
t_start = time()

# 第一阶段：异步传输数据到GPU
for k in range(num_arrays):
    # 将数据异步传输到GPU，与对应的流关联
    data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))

# 第二阶段：在GPU上执行核函数
for k in range(num_arrays):
    # 记录核函数开始事件
    start_events[k].record(streams[k])

    # 在指定流上执行核函数
    # block=(64,1,1)：每个块64个线程
    # grid=(1,1,1)：使用1个块
    mult_ker(data_gpu[k], np.int32(array_len), block=(
        64, 1, 1), grid=(1, 1, 1), stream=streams[k])

# 第三阶段：记录核函数结束事件
for k in range(num_arrays):
    # 记录核函数结束事件
    end_events[k].record(streams[k])

# 第四阶段：异步获取计算结果
for k in range(num_arrays):
    # 从GPU异步获取计算结果，与对应的流关联
    gpu_out.append(data_gpu[k].get_async(stream=streams[k]))

# 记录总体执行结束时间
t_end = time()

# 验证计算结果的正确性
for k in range(num_arrays):
    # 检查GPU计算结果是否与原始数据一致（考虑到浮点运算精度）
    assert (np.allclose(gpu_out[k], data[k]))

# 计算每个核函数的精确执行时间
kernel_times = []
for k in range(num_arrays):
    # 使用事件计算从开始到结束的精确时间（毫秒）
    kernel_times.append(start_events[k].time_till(end_events[k]))

# 输出性能统计信息
print("total time: %f" % (t_end-t_start))  # 总体执行时间
print("Mean kernel duration (milliseconds): %f" %
      np.mean(kernel_times))  # 平均内核执行时间
print("Mean kernel standard deviation (milliseconds): %f" %
      np.std(kernel_times))  # 内核执行时间的标准差
