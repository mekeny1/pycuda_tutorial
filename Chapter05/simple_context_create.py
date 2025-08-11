'''
Author: mekeny1
Date: 2025-06-13 01:31:10
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:51:10
FilePath: \pycuda_tutorial_hapril\Chapter05\simple_context_create.py
Description: 使用PyCUDA演示CUDA上下文(Context)的基本创建和管理，展示GPU设备初始化、上下文创建、内存操作和资源清理的完整流程
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv

"""
代码总体说明：
本程序演示了CUDA上下文的基本创建和管理，主要特点包括：

1. 算法思想：
   - 展示CUDA编程的基本流程：初始化→设备选择→上下文创建→操作执行→资源清理
   - 通过简单的数组传输和计算演示GPU的基本功能
   - 演示正确的CUDA资源管理方式

2. CUDA上下文管理：
   - 创建独立的CUDA上下文，管理GPU资源和状态
   - 在上下文中执行GPU操作，确保资源的正确分配和释放
   - 通过pop()方法正确销毁上下文，释放GPU资源

3. GPU操作流程：
   - 将CPU数据传输到GPU内存
   - 在GPU上执行基本操作
   - 将结果从GPU传输回CPU

4. 软硬件特性利用：
   - GPU：提供并行计算能力和专用内存空间
   - CPU：负责数据准备、GPU操作控制和结果处理
   - 内存：使用GPU全局内存进行数据传输和存储
   - 上下文：管理GPU资源状态和操作队列
"""

# 初始化CUDA驱动程序
# 这是使用PyCUDA进行GPU编程的第一步，必须在使用任何CUDA功能之前调用
drv.init()

# 选择GPU设备
# Device(0)表示选择系统中的第一个GPU设备（索引为0）
# 如果系统有多个GPU，可以通过不同的索引选择不同的设备
dev = drv.Device(0)

# 为选定的GPU设备创建CUDA上下文
# 上下文是GPU资源管理的基本单位，包含：
# - GPU内存分配
# - 核函数执行队列
# - 设备状态信息
# - 其他GPU资源
ctx = dev.make_context()

# 在GPU上下文中执行基本操作
# 将CPU上的numpy数组传输到GPU内存
# np.float32([1, 2, 3])：创建包含三个浮点数的数组
# gpuarray.to_gpu()：将数组传输到GPU全局内存
x = gpuarray.to_gpu(np.float32([1, 2, 3]))

# 从GPU获取数据并打印结果
# x.get()：将GPU内存中的数据同步传输回CPU
# 这里会显示原始数组[1. 2. 3.]，因为还没有进行任何计算
print(x.get())

# 销毁CUDA上下文，释放GPU资源
# ctx.pop()：弹出当前上下文，释放所有相关的GPU资源
# 这是CUDA编程中的重要步骤，确保GPU资源被正确释放
# 如果不调用pop()，GPU资源可能会泄漏，影响系统性能
ctx.pop()
