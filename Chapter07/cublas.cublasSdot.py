'''
Author: mekeny1
Date: 2025-06-24 16:33:44
LastEditors: mekeny1
LastEditTime: 2025-08-12 10:07:11
FilePath: \pycuda_tutorial_hapril\Chapter07\cublas.cublasSdot.py
Description: 
使用cuBLAS库演示GPU上的向量点积和L2范数计算
@algorithm: 向量点积运算和L2范数计算，基础线性代数运算
@cuda: 利用cuBLAS库的优化向量运算实现，单精度浮点运算
@demo: 展示cuBLAS向量内积和范数计算的GPU加速效果
@verification: 与NumPy计算结果对比，验证GPU计算的正确性
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas

# 定义测试向量和标量参数
a = np.float32(10)  # 标量参数（本示例中未使用）
v = np.float32([1, 2, 3])  # 第一个向量
w = np.float32([4, 5, 6])  # 第二个向量

# 将向量数据转移到GPU内存，启用并行计算
v_gpu = gpuarray.to_gpu(v)
w_gpu = gpuarray.to_gpu(w)

# 创建cuBLAS上下文句柄，管理GPU计算资源
# 这是cuBLAS库的标准初始化步骤
cublas_context_h = cublas.cublasCreate()

# 计算向量点积：dot(v, w) = v[0]*w[0] + v[1]*w[1] + v[2]*w[2]
# cublasSdot参数说明：
# - cublas_context_h: cuBLAS句柄
# - v_gpu.size: 向量长度
# - v_gpu.gpudata: 第一个向量的GPU内存地址
# - 1: 第一个向量的步长(stride)
# - w_gpu.gpudata: 第二个向量的GPU内存地址
# - 1: 第二个向量的步长(stride)
# 预期结果：1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
dot_output = cublas.cublasSdot(cublas_context_h, v_gpu.size,
                               v_gpu.gpudata, 1, w_gpu.gpudata, 1)

# 计算向量v的L2范数：||v||₂ = √(v[0]² + v[1]² + v[2]²)
# cublasSnrm2参数说明：
# - cublas_context_h: cuBLAS句柄
# - v_gpu.size: 向量长度
# - v_gpu.gpudata: 向量的GPU内存地址
# - 1: 向量的步长(stride)
# 预期结果：√(1² + 2² + 3²) = √(1 + 4 + 9) = √14 ≈ 3.7417
l2_output = cublas.cublasSnrm2(cublas_context_h, v_gpu.size, v_gpu.gpudata, 1)

# 销毁cuBLAS上下文句柄，释放GPU资源
# 这是cuBLAS库的标准清理步骤
cublas.cublasDestroy(cublas_context_h)

# 使用NumPy计算相同的运算，用于结果验证
numpy_dot = np.dot(v, w)  # NumPy点积计算
numpy_l2 = np.linalg.norm(v)  # NumPy L2范数计算

# 验证GPU计算结果与NumPy计算结果的正确性
# np.allclose比较两个数值是否在数值上接近（考虑浮点误差）
print("点积结果是否接近 NumPy 近似值: %s" % np.allclose(dot_output, numpy_dot))
print("L2 范数结果是否接近 NumPy 近似值: %s" % np.allclose(l2_output, numpy_l2))
