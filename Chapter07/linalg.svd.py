'''
Author: mekeny1
Date: 2025-06-26 00:46:29
LastEditors: mekeny1
LastEditTime: 2025-08-12 10:07:38
FilePath: \pycuda_tutorial_hapril\Chapter07\linalg.svd.py
Description: 
使用cuSOLVER在GPU上执行矩阵的奇异值分解(SVD)并验证重构
@algorithm: SVD分解 A = U * Σ * V^T，矩阵分解与重构
@cuda: 使用scikit-cuda封装的cuSOLVER进行GPU加速SVD
@verification: 通过np.allclose对SVD重构结果进行数值校验
@performance: 对大规模矩阵的分解在GPU上具有显著加速优势
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import misc, linalg
misc.init()  # 初始化scikit-cuda运行环境（涉及CUDA上下文/句柄）

# 定义矩阵尺寸：row×col 的随机矩阵，将进行SVD分解
row = 1000
col = 5000

# 生成单精度随机矩阵，使用float32以匹配多数GPU高效计算的数据类型
a = np.random.rand(row, col).astype(np.float32)
# 将矩阵传输到GPU显存，启用后续在GPU上的线性代数运算
a_gpu = gpuarray.to_gpu(a)

# 在GPU上执行奇异值分解：A = U * Σ * V^T
# U_d: 左奇异向量矩阵（row×row或row×k）
# s_d: 奇异值向量（非负，按降序排列）
# V_d: 右奇异向量矩阵（col×col或k×col）
# 通过lib="cusolver"使用NVIDIA cuSOLVER库的SVD实现
U_d, s_d, V_d = linalg.svd(a_gpu, lib="cusolver")

# 将GPU结果拷回CPU内存，便于后续重构与验证
U = U_d.get()
s = s_d.get()
V = V_d.get()

# 构造Σ矩阵（对角为奇异值），注意Σ为row×col的矩形对角阵
S = np.zeros((row, col))
S[:min(row, col), :min(row, col)] = np.diag(s)

# 使用U, Σ, V重构原矩阵，并与原始矩阵a进行数值近似比较
# 允许微小误差（atol=1e-5）以适配浮点与GPU计算差异
print("Can We reconstrut a from its SVD decomposition? :%s" %
      np.allclose(a, np.dot(U, np.dot(S, V)), atol=1e-5))
