'''
Author: mekeny1
Date: 2025-06-25 15:40:45
LastEditors: mekeny1
LastEditTime: 2025-08-12 10:07:23
FilePath: \pycuda_tutorial_hapril\Chapter07\fft.fft.py
Description: 
使用cuFFT库演示GPU上的快速傅里叶变换(FFT)运算
@algorithm: 快速傅里叶变换和逆变换，时域与频域转换
@cuda: 利用cuFFT库的优化FFT实现，支持实数到复数变换
@demo: 展示cuFFT正变换和逆变换的GPU加速效果
@verification: 与NumPy FFT计算结果对比，验证GPU计算的正确性
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft

# 定义FFT变换的长度
N = 1000  # 信号长度，选择1000便于FFT计算（接近2的幂次）

# 生成随机实数信号作为输入数据
x = np.asarray(np.random.rand(N), dtype=np.float32)
x_gpu = gpuarray.to_gpu(x)  # 将实数信号转移到GPU内存

# 在GPU上分配复数输出数组，用于存储FFT结果
# cuFFT将实数输入转换为复数输出，因此输出类型为complex64
x_hat = gpuarray.empty_like(x_gpu, dtype=np.complex64)

# 创建FFT计划，优化GPU内存访问和计算性能
# 输入类型：float32（实数），输出类型：complex64（复数）
plan = fft.Plan(x_gpu.shape, np.float32, np.complex64)

# 创建IFFT（逆FFT）计划，用于从频域转换回时域
# 输入类型：complex64（复数），输出类型：float32（实数）
inverse_plan = fft.Plan(x.shape, in_dtype=np.complex64, out_dtype=np.float32)

# 执行正向FFT：将时域信号x转换为频域信号x_hat
# fft.fft函数将实数输入转换为复数输出
fft.fft(x_gpu, x_hat, plan)

# 执行逆FFT：将频域信号x_hat转换回时域信号x_gpu
# scale=True参数应用缩放因子1/N，确保逆变换的正确性
fft.ifft(x_hat, x_gpu, inverse_plan, scale=True)

# 使用NumPy计算相同的FFT，用于结果验证
y = np.fft.fft(x)

# cuFFT在处理实数时，只计算前半部分的输出，其余部分设为0，而NumPy不会
# 这是因为实数信号的FFT具有共轭对称性，后半部分是前半部分的共轭
# 因此只需要比较前半部分的结果即可验证正确性
# print("cuFFT matches Numpy FFT: %s" % np.allclose(x_hat.get(), y, atol=1e-6))
print("cuFFT matches Numpy FFT: %s" % np.allclose(
    x_hat.get()[0:N//2], y[0:N//2], atol=1e-6))

# 验证逆FFT结果是否与原始信号一致
# 由于数值精度误差，使用atol=1e-6的容差进行比较
print("cuFFT inverse matches original: %s" %
      np.allclose(x_gpu.get(), x, atol=1e-6))
