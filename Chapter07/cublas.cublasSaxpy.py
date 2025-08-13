'''
Author: mekeny1
Date: 2025-06-23 20:40:37
LastEditors: mekeny1
LastEditTime: 2025-08-12 10:07:06
FilePath: \pycuda_tutorial_hapril\Chapter07\cublas.cublasSaxpy.py
Description: 
使用cuBLAS库演示GPU上的向量加法运算(SAXPY)
@algorithm: SAXPY操作 y = α*x + y，向量缩放加法运算
@cuda: 利用cuBLAS库的优化SAXPY实现，单精度浮点运算
@demo: 展示cuBLAS基本向量运算的GPU加速效果
@verification: 与NumPy计算结果对比，验证GPU计算的正确性
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas

# 定义SAXPY操作的参数：y = α*x + y
# α: 标量缩放因子
a = np.float32(10)
# x: 输入向量
x = np.float32([1, 2, 3])
# y: 输入/输出向量（将被修改）
y = np.float32([-.345, 8.15, -15.867])

# 将向量数据转移到GPU内存，启用并行计算
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)

# 创建cuBLAS上下文句柄，管理GPU计算资源
# 这是cuBLAS库的标准初始化步骤
cublas_context_h = cublas.cublasCreate()

# 执行SAXPY操作：y = α*x + y
# cublasSaxpy参数说明：
# - cublas_context_h: cuBLAS句柄
# - x_gpu.size: 向量长度
# - a: 标量缩放因子α
# - x_gpu.gpudata: 输入向量x的GPU内存地址
# - 1: x向量的步长(stride)
# - y_gpu.gpudata: 输出向量y的GPU内存地址
# - 1: y向量的步长(stride)
cublas.cublasSaxpy(cublas_context_h, x_gpu.size, a,
                   x_gpu.gpudata, 1, y_gpu.gpudata, 1)

# 销毁cuBLAS上下文句柄，释放GPU资源
# 这是cuBLAS库的标准清理步骤
cublas.cublasDestroy(cublas_context_h)

# 验证GPU计算结果与NumPy计算结果的正确性
# np.allclose比较两个数组是否在数值上接近（考虑浮点误差）
# 预期结果：y = 10*[1,2,3] + [-0.345, 8.15, -15.867] = [9.655, 28.15, 14.133]
print("This is close to the Numpy approximation: %s" %
      np.allclose(a*x+y, y_gpu.get()))
