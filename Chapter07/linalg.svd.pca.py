'''
Author: mekeny1
Date: 2025-06-26 01:20:34
LastEditors: mekeny1
LastEditTime: 2025-08-12 10:07:28
FilePath: \pycuda_tutorial_hapril\Chapter07\linalg.svd.pca.py
Description: 
使用cuSOLVER库演示GPU上的奇异值分解(SVD)运算，用于主成分分析(PCA)
@algorithm: 奇异值分解 A = U*Σ*V^T，矩阵分解算法
@cuda: 利用cuSOLVER库的优化SVD实现，支持大规模矩阵分解
@application: 主成分分析(PCA)，数据降维和特征提取
@performance: GPU并行计算显著提升大规模矩阵SVD性能
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import misc, linalg
misc.init()  # 初始化scikit-cuda库

# 创建模拟数据集，用于演示PCA和SVD
# 定义两个主要的主成分方向（特征向量）
vals = [np.float32([10, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 第一个主成分：沿第一个维度
        np.float32([0, 10, 0, 0, 0, 0, 0, 0, 0, 0])]   # 第二个主成分：沿第二个维度

# 生成3000个数据点，模拟真实数据集
for i in range(3000):
    # 添加围绕第一个主成分的噪声数据点
    vals.append(vals[0]+0.001*np.random.rand(10))
    # 添加围绕第二个主成分的噪声数据点
    vals.append(vals[1]+0.001*np.random.rand(10))
    # 添加随机噪声数据点
    vals.append(0.001*np.random.rand(10))

# 将数据转换为numpy数组，使用单精度浮点数
vals = np.float32(vals)

# 数据预处理：中心化（减去均值）
# 这是PCA的标准预处理步骤，确保数据围绕原点分布
vals = vals-np.mean(vals, axis=0)

# 将数据转移到GPU内存，使用转置格式（列主序）
# 转置是为了符合cuSOLVER的输入要求
v_gpu = gpuarray.to_gpu(vals.T.copy())

# 执行奇异值分解：A = U*Σ*V^T
# 使用cuSOLVER库进行GPU加速的SVD计算
# U_d: 左奇异向量矩阵（正交矩阵）
# s_d: 奇异值向量（对角矩阵Σ的对角元素）
# V_d: 右奇异向量矩阵（正交矩阵）
U_d, s_d, V_d = linalg.svd(v_gpu, lib="cusolver")

# 将GPU计算结果传输回CPU内存
u = U_d.get()  # 左奇异向量矩阵
s = s_d.get()  # 奇异值向量
v = V_d.get()  # 右奇异向量矩阵
