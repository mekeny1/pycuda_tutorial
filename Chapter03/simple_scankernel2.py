'''
Author: mekeny1
Date: 2025-05-27 22:33:22
LastEditors: mekeny1
LastEditTime: 2025-08-11 01:38:58
FilePath: \pycuda_tutorial_hapril\Chapter03\simple_scankernel2.py
Description: PyCUDA前缀和算法实现最大值查找 - InclusiveScanKernel归约操作示例
result:
# 10000
# 10000
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel

"""
PyCUDA前缀和算法实现最大值查找程序

该程序演示了如何使用InclusiveScanKernel实现数组最大值查找。
通过将二元操作符设置为最大值比较，前缀和算法可以高效地找到数组中的最大值。
这是并行归约操作的一个典型应用。

算法处理流程：
1. 创建包含正负数的测试数组
2. 将数组传输到GPU内存
3. 使用InclusiveScanKernel计算最大值前缀和
4. 取结果的最后一个元素作为全局最大值
5. 与CPU的max函数结果进行对比验证

核心方法：
- InclusiveScanKernel: 使用自定义二元操作符的前缀和内核
- "a>b?a:b": 三元操作符，实现最大值比较
- kernel.get()[-1]: 获取前缀和结果的最后一个元素
- np.max(): NumPy的CPU最大值查找函数

CUDA相关概念：
- 归约操作：将数组归约为单个值（如最大值、最小值、总和等）
- 前缀和变体：通过改变二元操作符实现不同的归约功能
- 并行归约：使用并行算法高效计算归约结果
- 扫描算法：前缀和算法的通用形式，支持任意二元操作

软硬件特性：
- 并行归约：GPU同时处理多个元素对的最大值比较
- 内存访问：归约操作需要高效的内存访问模式
- 算法复杂度：并行归约的复杂度为O(log n)
- 数值稳定性：处理包含正负数的数组
"""

# 创建测试数组：包含正负数的整数序列
# 数组包含各种数值，用于测试最大值查找的正确性
seq = np.array([1, 100, -3, -1000, 4, 10000, 66, 14, 21], dtype=np.int32)

# 将数组从CPU传输到GPU内存
# 这是GPU计算的第一步：数据准备
seq_gpu = gpuarray.to_gpu(seq)

# 创建最大值前缀和计算内核
# InclusiveScanKernel参数说明：
# - np.int32: 数据类型，指定输入和输出的数据类型
# - "a>b?a:b": 三元操作符，实现最大值比较
#   如果a大于b，返回a；否则返回b
# 这个操作会将数组转换为最大值前缀和：
# [1,100,-3,-1000,4,10000,66,14,21] -> [1,100,100,100,100,10000,10000,10000,10000]
max_gpu = InclusiveScanKernel(np.int32, "a>b?a:b")

# 在GPU上执行最大值前缀和计算
# 内核函数自动处理并行计算，无需手动管理线程
gpu_result = max_gpu(seq_gpu)

# 获取GPU计算结果的最后一个元素作为全局最大值
# 前缀和的最后一个元素包含了整个数组的最大值
print(gpu_result.get()[-1])

# 使用NumPy的max函数在CPU上查找最大值
# 用于验证GPU计算结果的正确性
cpu_result = np.max(seq)
print(cpu_result)
