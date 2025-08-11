'''
Author: mekeny1
Date: 2025-05-23 00:33:29
LastEditors: mekeny1
LastEditTime: 2025-08-11 01:42:37
FilePath: \pycuda_tutorial_hapril\Chapter03\time_calc0.py
Description: PyCUDA CPU与GPU性能对比测试 - 数组乘法运算性能分析
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time

"""
PyCUDA CPU与GPU性能对比测试程序

该程序通过数组乘法运算（每个元素乘以2）对比CPU和GPU的计算性能。
使用5000万个float32随机数作为测试数据，测量计算时间差异。

算法处理流程：
1. 生成5000万个随机浮点数
2. 在CPU上执行数组乘法并记录时间
3. 将数据传输到GPU
4. 在GPU上执行相同乘法并记录时间
5. 将结果传回CPU并验证正确性

核心方法：
- gpuarray.to_gpu(): 数据传输到GPU
- gpuarray乘法: GPU上的自动并行计算
- kernel.get(): 结果传回CPU
- np.allclose(): 验证计算结果一致性
"""

# 生成大规模测试数据：5000万个随机浮点数
# 使用float32类型以匹配GPU计算精度，减少内存占用
host_data = np.float32(np.random.random(50000000))

# CPU计算部分
# 记录CPU计算开始时间
t1 = time()
# 在CPU上执行数组乘法：每个元素乘以2
# NumPy的向量化运算在CPU上执行
host_data_2x = host_data*np.float32(2)
# 记录CPU计算结束时间
t2 = time()
# 输出CPU计算耗时
print("CPU计算总耗时: %f 秒" % (t2-t1))

# GPU计算部分
# 将数据从CPU传输到GPU内存
# 这是GPU计算的第一步：数据准备
device_data = gpuarray.to_gpu(host_data)

# 记录GPU计算开始时间
t1 = time()
# 在GPU上执行数组乘法：每个元素乘以2
# PyCUDA自动处理并行计算，无需手动编写内核函数
# GPU数组运算会自动并行执行，每个CUDA核心处理多个元素
device_data_2x = device_data*np.float32(2)
# 记录GPU计算结束时间
t2 = time()

# 将GPU计算结果传输回CPU内存
# 这是GPU计算的最后一步：结果获取
from_device = device_data_2x.get()

# 输出GPU计算耗时
print('GPU计算总耗时: %f 秒' % (t2 - t1))

# 验证GPU计算结果与CPU结果是否一致
# np.allclose()检查两个数组是否在数值精度范围内相等
# 考虑到浮点运算的精度差异，使用相对误差容限进行比较
print('GPU计算结果与CPU结果是否一致: {}'.format(
    np.allclose(from_device, host_data_2x)))
