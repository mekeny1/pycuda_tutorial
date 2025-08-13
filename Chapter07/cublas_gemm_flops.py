'''
Author: mekeny1
Date: 2025-06-25 02:19:40
LastEditors: mekeny1
LastEditTime: 2025-08-12 10:25:04
FilePath: \pycuda_tutorial_hapril\Chapter07\cublas_gemm_flops.py
Description: 
使用cuBLAS库测试GPU上矩阵乘法的浮点运算性能
@algorithm: 通用矩阵乘法(GEMM) C = α*A*B + β*C
@cuda: 利用cuBLAS库的优化GEMM实现，支持单精度和双精度运算
@performance: 测量GPU的GFLOPS性能，评估计算吞吐量
@benchmark: 通过大规模矩阵乘法测试GPU的理论峰值性能
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas
from time import time

# 定义测试矩阵的维度：A(m×k) * B(k×n) = C(m×n)
# 选择大规模矩阵以充分测试GPU性能
m = 5000
n = 10000
k = 10000


def compute_gflops(precision='S'):
    """
    计算GPU上矩阵乘法的GFLOPS性能

    算法：C = α*A*B + β*C (GEMM操作)
    性能计算：GFLOPS = (2*m*n*k) / (执行时间 * 10^9)

    核心优化技术：
    1. 使用cuBLAS库的高度优化GEMM实现
    2. 列主序存储格式优化内存访问模式
    3. GPU并行计算充分利用多核架构

    Args:
        precision: 精度类型，'S'表示单精度(float32)，'D'表示双精度(float64)

    Returns:
        GPU的GFLOPS性能值
    """
    if precision == 'S':
        float_type = "float32"
    elif precision == 'D':
        float_type = "float64"
    else:
        return -1

    # 生成随机测试矩阵，使用指定精度
    A = np.random.rand(m, k).astype(float_type)
    B = np.random.rand(k, n).astype(float_type)
    C = np.random.rand(m, n).astype(float_type)

    # 转换为列主序格式，cuBLAS使用列主序存储
    # 这是cuBLAS库的标准要求，确保最佳性能
    A_cm = A.T.copy()
    B_cm = B.T.copy()
    C_cm = C.T.copy()

    # 将数据转移到GPU内存，启用并行计算
    A_gpu = gpuarray.to_gpu(A_cm)
    B_gpu = gpuarray.to_gpu(B_cm)
    C_gpu = gpuarray.to_gpu(C_cm)

    # 设置GEMM操作的标量参数
    alpha = np.random.rand()  # 缩放因子α
    beta = np.random.rand()   # 缩放因子β

    # 设置矩阵转置标志：'N'表示不转置
    transa = cublas._CUBLAS_OP['N']
    transb = cublas._CUBLAS_OP['N']

    # 设置矩阵的leading dimension（列主序下的行数）
    lda = m  # A矩阵的leading dimension
    ldb = k  # B矩阵的leading dimension
    ldc = m  # C矩阵的leading dimension

    # 开始性能测试计时
    t = time()

    # 创建cuBLAS句柄，管理GPU计算资源
    handle = cublas.cublasCreate()

    # 执行GEMM操作：C = α*A*B + β*C
    # 使用动态函数调用支持不同精度类型
    # cublasSgemm: 单精度，cublasDgemm: 双精度
    exec("cublas.cublas%sgemm(handle, transa, transb, m, n, k, alpha, A_gpu.gpudata, lda, B_gpu.gpudata, ldb, beta, C_gpu.gpudata, ldc)" % precision)

    # 销毁cuBLAS句柄，释放GPU资源
    cublas.cublasDestroy(handle)

    # 计算执行时间
    t = time()-t

    # 计算GFLOPS：浮点运算次数 / (执行时间 * 10^9)
    # GEMM的浮点运算次数：2*m*n*k（每个输出元素需要k次乘法和k-1次加法）
    # 额外加上m*n次β*C的运算，总计2*m*n*(k+1)
    gflops = 2*m*n*(k+1)*(10**9)/t
    return gflops


if __name__ == "__main__":
    # 主程序：测试单精度和双精度GEMM性能
    # 单精度通常比双精度快2-4倍，但精度较低
    print("Single-precision performance: %s GFLOPS" % compute_gflops('S'))
    print("Double-precision performance: %s GFLOPS" % compute_gflops('D'))
