'''
Author: mekeny1
Date: 2025-06-13 21:36:41
LastEditors: mekeny1
LastEditTime: 2025-08-11 12:15:49
FilePath: \pycuda_tutorial_hapril\Chapter05\multi-kernel_multi-thread.py
Description: 使用PyCUDA演示多核函数多线程并发执行，通过Python多线程管理多个独立的CUDA上下文，实现CPU多线程与GPU并行计算的结合
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import threading

"""
代码总体说明：
本程序演示了多核函数多线程并发执行的实现，主要特点包括：

1. 算法思想：
   - 使用Python多线程创建多个独立的CUDA上下文
   - 每个线程管理一个独立的GPU设备和上下文
   - 通过多线程实现真正的并行GPU计算

2. GPU并行化策略：
   - 每个线程在独立的CUDA上下文中执行核函数
   - 避免了CUDA上下文的竞争和同步问题
   - 实现了真正的多GPU并行计算（如果系统支持）

3. 多线程与CUDA结合：
   - Python多线程负责管理多个CUDA上下文
   - 每个线程独立编译和执行CUDA核函数
   - 通过线程join实现结果收集和同步

4. 软硬件特性利用：
   - GPU：每个线程使用独立的CUDA上下文，避免资源竞争
   - CPU：多线程管理，提高CPU利用率
   - 内存：每个线程独立管理GPU内存，避免内存冲突
   - 上下文管理：独立的CUDA上下文确保计算隔离
"""

# 程序参数设置
num_arrays = 10  # 并发处理的数组数量
array_len = 1024**2  # 每个数组的长度（1M个元素）

# 定义CUDA核函数代码，执行数组的重复乘除运算
# 这个核函数主要用于演示GPU计算和多线程管理
kernel_code = """
__global__ void mult_ker(float *array, int array_len)
{
    int thd = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程的全局索引
    int num_iters = array_len / blockDim.x;  // 计算每个线程需要处理的迭代次数

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


class KernelLauncherThread(threading.Thread):
    """
    内核启动器线程类，继承自Python的Thread类

    每个线程实例负责：
    1. 创建独立的CUDA上下文
    2. 编译和执行CUDA核函数
    3. 管理GPU内存和计算资源
    """

    def __init__(self, input_array):
        """
        初始化线程

        参数：
        - input_array: 输入数组，将在GPU上处理
        """
        threading.Thread.__init__(self)  # 调用父类构造函数
        self.input_array = input_array  # 存储输入数组
        self.output_array = None  # 存储输出数组

    def run(self):
        """
        线程的主要执行方法，包含完整的GPU计算流程
        """
        # 获取GPU设备（使用设备0）
        self.dev = drv.Device(0)

        # 为当前线程创建独立的CUDA上下文
        # 这确保了每个线程有独立的GPU资源管理
        self.context = self.dev.make_context()

        # 在独立上下文中编译CUDA核函数代码
        self.ker = SourceModule(kernel_code)

        # 从编译后的模块中获取核函数
        self.mult_ker = self.ker.get_function("mult_ker")

        # 将输入数组传输到GPU内存
        self.array_gpu = gpuarray.to_gpu(self.input_array)

        # 在GPU上执行核函数
        # block=(64,1,1)：每个块64个线程
        # grid=(1,1,1)：使用1个块
        self.mult_ker(self.array_gpu, np.int32(array_len),
                      block=(64, 1, 1), grid=(1, 1, 1))

        # 从GPU获取计算结果
        self.output_array = self.array_gpu.get()

        # 弹出当前CUDA上下文，释放GPU资源
        # 这是CUDA上下文管理的重要步骤
        self.context.pop()

    def join(self):
        """
        重写join方法，返回计算结果

        返回：
        - output_array: GPU计算的结果数组
        """
        threading.Thread.join(self)  # 调用父类join方法等待线程完成
        return self.output_array  # 返回计算结果


# 初始化CUDA驱动程序
drv.init()

# 初始化数据结构
data = []  # 存储CPU端的原始数据
gpu_out = []  # 存储GPU计算结果
threads = []  # 存储线程对象

# 生成随机测试数据
for _ in range(num_arrays):
    # 创建随机浮点数数组，用于GPU计算测试
    data.append(np.random.randn(array_len).astype("float32"))

# 创建多个内核启动器线程
for k in range(num_arrays):
    # 为每个数组创建一个独立的线程，传入对应的数据
    threads.append(KernelLauncherThread(data[k]))

# 启动所有线程，开始并发执行
for k in range(num_arrays):
    threads[k].start()  # 启动线程，开始GPU计算

# 等待所有线程完成并收集结果
for k in range(num_arrays):
    # 等待线程完成并获取计算结果
    gpu_out.append(threads[k].join())

# 验证计算结果的正确性
for k in range(num_arrays):
    # 检查GPU计算结果是否与原始数据一致（考虑到浮点运算精度）
    assert (np.allclose(gpu_out[k], data[k]))
