'''
Author: mekeny1
Date: 2025-06-12 01:47:58
LastEditors: mekeny1
LastEditTime: 2025-08-11 12:24:20
FilePath: \pycuda_tutorial_hapril\Chapter05\simple_event_example.py
Description: 使用PyCUDA演示CUDA事件(Events)的基本使用，展示如何通过事件机制精确测量GPU核函数的执行时间，实现GPU性能监控和同步控制
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as dvr
from time import time

"""
代码总体说明：
本程序演示了CUDA事件的基本使用，主要特点包括：

1. 算法思想：
   - 使用CUDA事件记录GPU核函数的开始和结束时间点
   - 通过事件查询和同步机制实现精确的时间测量
   - 展示事件在GPU性能监控和同步控制中的作用

2. GPU性能测量策略：
   - 在核函数执行前后记录事件时间戳
   - 使用事件查询检查核函数的执行状态
   - 通过事件时间差计算精确的内核执行时间

3. CUDA事件机制：
   - 事件是GPU操作的时间标记点，用于同步和性能测量
   - 支持异步记录，不阻塞CPU执行
   - 提供查询和同步功能，实现精确的时间控制

4. 软硬件特性利用：
   - GPU：提供高精度的硬件时间戳和事件记录能力
   - CPU：负责事件管理、状态查询和结果分析
   - 内存：使用GPU全局内存存储测试数据
   - 事件系统：实现GPU操作的精确时间测量和同步控制
"""

# 定义CUDA核函数，执行数组的重复乘除运算
# 这个核函数主要用于演示GPU计算和事件测量
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
array_len = 100*1024**2  # 数组长度：100MB（100M个浮点数）
data = np.random.randn(array_len).astype("float32")  # 生成随机测试数据

# 将数据传输到GPU内存
data_gpu = gpuarray.to_gpu(data)

# 创建CUDA事件对象，用于时间测量
start_event = dvr.Event()  # 开始事件，记录核函数开始执行的时间点
end_event = dvr.Event()    # 结束事件，记录核函数完成执行的时间点

# 记录核函数开始执行的事件
# record()方法在GPU上记录当前时间点，不阻塞CPU执行
start_event.record()

# 在GPU上执行核函数
# block=(64,1,1)：每个块64个线程
# grid=(1,1,1)：使用1个块
mult_ker(data_gpu, np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1))

# 记录核函数完成执行的事件
# record()方法在GPU上记录当前时间点，不阻塞CPU执行
end_event.record()

# 同步等待结束事件完成
# synchronize()确保end_event已经被记录，CPU会等待直到事件完成
end_event.synchronize()

# 查询事件状态，检查核函数的执行情况
# query()方法返回布尔值，表示事件是否已经完成
print("Has the kernel started yet? {}".format(start_event.query()))  # 检查开始事件状态
print("Has the kernel ended yet? {}".format(end_event.query()))      # 检查结束事件状态

# 计算核函数的精确执行时间
# time_till()方法计算从start_event到end_event的时间差（毫秒）
# 这是CUDA事件最重要的功能：精确的时间测量
print("Kernel execution time in milliseconds: %f" %
      start_event.time_till(end_event))
