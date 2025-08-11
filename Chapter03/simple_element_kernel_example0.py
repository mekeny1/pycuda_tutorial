'''
Author: mekeny1
Date: 2025-05-25 17:01:59
LastEditors: mekeny1
LastEditTime: 2025-08-11 01:30:00
FilePath: \pycuda_tutorial_hapril\Chapter03\simple_element_kernel_example0.py
Description: PyCUDA ElementwiseKernel基础示例 - CPU与GPU性能对比
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel

"""
PyCUDA ElementwiseKernel基础示例程序

该程序演示了PyCUDA ElementwiseKernel的基本使用方法，通过简单的数组乘法运算
对比CPU和GPU的计算性能。这是学习GPU并行编程的入门示例。

算法处理流程：
1. 生成大规模随机数据（5000万个浮点数）
2. 在CPU上执行数组乘法运算（每个元素乘以2）
3. 将数据传输到GPU内存
4. 在GPU上并行执行相同的乘法运算
5. 将GPU结果传输回CPU并验证正确性
6. 对比CPU和GPU的计算时间

核心方法：
- ElementwiseKernel: PyCUDA的元素级并行计算内核
- gpuarray.to_gpu(): CPU到GPU的数据传输
- gpuarray.empty_like(): 在GPU上分配相同形状的内存
- kernel.get(): GPU到CPU的结果传输

CUDA相关概念：
- 元素级并行：每个GPU线程处理数组中的一个元素
- 内存管理：GPU内存的分配和释放
- 数据传输：CPU和GPU之间的数据移动开销
- 内核函数：在GPU上并行执行的CUDA C代码

软硬件特性：
- 大规模并行：GPU同时处理数百万个数据元素
- 内存带宽：GPU内存带宽远高于CPU，适合数据密集型计算
- 计算密度：GPU在简单重复计算上具有显著优势
- 数据传输开销：CPU-GPU数据传输可能成为性能瓶颈
"""

# 生成大规模测试数据：5000万个随机浮点数
# 使用float32类型以匹配GPU计算精度
host_data = np.float32(np.random.random(50000000))

# 定义GPU内核函数 - 使用ElementwiseKernel
# 每个线程处理一个数组元素，执行简单的乘法运算
gpu_2x_ker = ElementwiseKernel(
    # 参数列表：输入数组指针、输出数组指针
    "float *in, float *out",
    # CUDA C代码：每个线程将输入元素乘以2并存储到输出
    "out[i] = in[i] * 2;",
    # 内核函数名称（CUDA C命名空间）
    "gpu_2x_ker"
)


def speedcomparison():
    """
    CPU与GPU性能对比函数

    执行流程：
    1. 在CPU上执行数组乘法
    2. 在GPU上执行相同的运算
    3. 对比计算时间和结果正确性

    性能分析：
    - CPU：串行执行，受限于单核性能
    - GPU：并行执行，数千个线程同时工作
    - 数据传输：CPU-GPU之间的数据移动开销
    """
    # CPU计算部分
    t1 = time()
    # 在CPU上执行数组乘法：每个元素乘以2
    host_data_2x = host_data * np.float32(2)
    t2 = time()
    print("CPU计算总耗时: %f 秒" % (t2-t1))

    # GPU计算部分
    # 将数据从CPU传输到GPU内存
    device_data = gpuarray.to_gpu(host_data)

    # 在GPU上分配输出数组内存
    # empty_like()创建与输入数组相同形状和类型的内存空间
    # 内核函数涉及C语言中的浮点数类型指针，所以此处需提前分配内存
    device_data_2x = gpuarray.empty_like(device_data)

    # 记录GPU计算开始时间
    t1 = time()
    # 在GPU上并行执行内核函数
    # 每个GPU线程处理一个数组元素，同时进行乘法运算
    gpu_2x_ker(device_data, device_data_2x)
    t2 = time()

    # 将GPU计算结果传输回CPU内存
    from_device = device_data_2x.get()

    # 输出GPU计算时间
    print("GPU计算总耗时: %f 秒" % (t2 - t1))

    # 验证GPU计算结果与CPU结果是否一致
    # np.allclose()检查两个数组是否在数值精度范围内相等
    print("GPU计算结果与CPU结果是否一致: {}".format(
        np.allclose(from_device, host_data_2x)))


if __name__ == "__main__":
    # 执行CPU与GPU性能对比测试
    speedcomparison()
