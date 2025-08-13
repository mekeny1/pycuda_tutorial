'''
Author: mekeny1
Date: 2025-06-25 01:30:08
LastEditors: mekeny1
LastEditTime: 2025-08-12 10:07:19
FilePath: \pycuda_tutorial_hapril\Chapter07\cublas.cublasSgemv.py
Description: 
使用cuBLAS库演示GPU上的矩阵-向量乘法运算(GEMV)
@algorithm: 通用矩阵-向量乘法 y = α*A*x + β*y，线性变换运算
@cuda: 利用cuBLAS库的优化GEMV实现，单精度浮点运算
@demo: 展示cuBLAS矩阵-向量乘法的GPU加速效果
@verification: 与NumPy计算结果对比，验证GPU计算的正确性
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas

# 定义GEMV操作的矩阵和向量维度
m = 10  # 矩阵A的行数，输出向量y的长度
n = 100  # 矩阵A的列数，输入向量x的长度
alpha = 1  # 缩放因子α
beta = 0   # 缩放因子β

# 生成测试数据
A = np.random.rand(m, n).astype("float32")  # 随机矩阵A (m×n)
x = np.random.rand(n).astype("float32")     # 输入向量x (n×1)
y = np.zeros(m).astype("float32")           # 输出向量y (m×1)，初始化为零

# 将矩阵A转换为列主序格式，cuBLAS使用列主序存储
# A.T.copy()创建转置矩阵的副本，确保内存布局符合cuBLAS要求
A_columnwise = A.T.copy()
A_gpu = gpuarray.to_gpu(A_columnwise)
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)

# 设置矩阵转置标志：'N'表示不转置矩阵A
# 其他选项：'T'表示转置，'C'表示共轭转置
trans = cublas._CUBLAS_OP['N']

# 设置cuBLAS GEMV操作的参数
lda = m    # 矩阵A的leading dimension（列主序下的行数）
incx = 1   # 向量x的步长(stride)
incy = 1   # 向量y的步长(stride)

# 创建cuBLAS句柄，管理GPU计算资源
handle = cublas.cublasCreate()

# 执行GEMV操作：y = α*A*x + β*y
# cublasSgemv参数说明：
# - handle: cuBLAS句柄
# - trans: 矩阵转置标志
# - m: 矩阵A的行数
# - n: 矩阵A的列数
# - alpha: 标量缩放因子α
# - A_gpu.gpudata: 矩阵A的GPU内存地址
# - lda: 矩阵A的leading dimension
# - x_gpu.gpudata: 输入向量x的GPU内存地址
# - incx: 向量x的步长
# - beta: 标量缩放因子β
# - y_gpu.gpudata: 输出向量y的GPU内存地址
# - incy: 向量y的步长
cublas.cublasSgemv(handle, trans, m, n, alpha, A_gpu.gpudata,
                   lda, x_gpu.gpudata, incx, beta, y_gpu.gpudata, incy)

# 销毁cuBLAS上下文句柄，释放GPU资源
cublas.cublasDestroy(handle)

# 验证GPU计算结果与NumPy计算结果的正确性
# 由于β=0，实际计算为：y = A*x
# np.allclose比较两个数组是否在数值上接近（考虑浮点误差）
print("cuBLAS returned the correct value: %s" %
      np.allclose(np.dot(A, x), y_gpu.get()))
