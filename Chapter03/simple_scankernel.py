'''
Author: mekeny1
Date: 2025-05-26 01:34:30
LastEditors: mekeny1
LastEditTime: 2025-08-11 01:35:08
FilePath: \pycuda_tutorial_hapril\Chapter03\simple_scankernel.py
Description: PyCUDA前缀和（Scan）算法示例 - InclusiveScanKernel使用演示
result:
# [ 1  3  6 10]
# [ 1  3  6 10]
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
# 自动初始化 CUDA 环境
import pycuda.autoinit
# 提供 GPU 上的数组操作
from pycuda import gpuarray
# 用于创建 GPU 上的前缀和计算内核
from pycuda.scan import InclusiveScanKernel

"""
PyCUDA前缀和（Scan）算法示例程序

该程序演示了PyCUDA的InclusiveScanKernel使用方法，实现数组的前缀和计算。
前缀和是并行计算中的基础算法，广泛应用于数据压缩、图像处理、数值积分等领域。

算法处理流程：
1. 创建输入数组 [1, 2, 3, 4]
2. 将数组传输到GPU内存
3. 使用InclusiveScanKernel计算包含性前缀和
4. 将结果从GPU传输回CPU
5. 与CPU的cumsum函数结果进行对比验证

核心方法：
- InclusiveScanKernel: PyCUDA提供的包含性前缀和计算内核
- gpuarray.to_gpu(): CPU到GPU的数据传输
- kernel.get(): GPU到CPU的结果传输
- np.cumsum(): NumPy的CPU前缀和函数

CUDA相关概念：
- 前缀和（Scan）：将数组转换为累积和的形式
- 包含性前缀和：每个输出位置包含当前元素及之前所有元素的和
- 并行扫描：使用并行算法高效计算前缀和
- 归约操作：将多个元素组合成单个结果的运算

软硬件特性：
- 并行计算：GPU同时处理多个数据元素
- 内存访问模式：前缀和算法需要高效的内存访问
- 算法复杂度：并行前缀和的复杂度为O(log n)
- 数据依赖：前缀和计算存在数据依赖关系，需要特殊的并行策略
"""

# 创建测试数组：整数序列 [1, 2, 3, 4]
# 使用int32类型以匹配GPU计算精度
seq = np.array([1, 2, 3, 4], dtype=np.int32)

# 将数组从CPU传输到GPU内存
# 这是GPU计算的第一步：数据准备
seq_gpu = gpuarray.to_gpu(seq)

# 创建包含性前缀和计算内核
# InclusiveScanKernel参数说明：
# - np.int32: 数据类型，指定输入和输出的数据类型
# - "a+b": 二元操作符，定义如何组合两个元素（这里是加法）
# 包含性前缀和：每个输出位置包含当前元素及之前所有元素的和
# 例如：[1,2,3,4] -> [1,1+2,1+2+3,1+2+3+4] = [1,3,6,10]
sum_gpu = InclusiveScanKernel(np.int32, "a+b")

# 在GPU上执行前缀和计算
# 内核函数自动处理并行计算，无需手动管理线程
gpu_result = sum_gpu(seq_gpu)

# 将GPU计算结果传输回CPU并显示
print(gpu_result.get())

# 使用NumPy的cumsum函数在CPU上计算相同的前缀和
# 用于验证GPU计算结果的正确性
cpu_result = np.cumsum(seq)
print(cpu_result)
