'''
Author: mekeny1
Date: 2025-06-25 16:43:33
LastEditors: mekeny1
LastEditTime: 2025-08-12 10:09:22
FilePath: \pycuda_tutorial_hapril\Chapter07\conv_2d.py
Description: 
使用PyCUDA和cuFFT库实现2D卷积运算的GPU加速程序
@algorithm: 基于FFT的快速卷积算法，利用频域乘法代替时域卷积
@cuda: 使用cuFFT进行GPU上的快速傅里叶变换，cuBLAS进行复数乘法运算
@performance: GPU并行计算显著提升大规模图像卷积性能
@application: 高斯滤波、图像模糊等图像处理任务
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from __future__ import division
import pycuda.autoinit
from pycuda import gpuarray

import numpy as np
from skcuda import fft, linalg
from matplotlib import pyplot as plt


def cufft_conv(x, y):
    """
    使用cuFFT库在GPU上执行快速卷积运算

    算法原理：利用卷积定理，时域卷积等价于频域乘法
    1. 将输入信号转换到频域（FFT）
    2. 在频域执行复数乘法
    3. 将结果转换回时域（IFFT）

    Args:
        x: 第一个复数数组（卷积核）
        y: 第二个复数数组（输入信号）

    Returns:
        卷积结果，如果输入形状不匹配则返回-1
    """
    x = x.astype(np.complex64)
    y = y.astype(np.complex64)

    if (x.shape != y.shape):
        return -1

    # 创建FFT和IFFT计划，优化GPU内存访问和计算性能
    plan = fft.Plan(x.shape, np.complex64, np.complex64)
    inverse_plan = fft.Plan(x.shape, np.complex64, np.complex64)

    # 将数据转移到GPU内存，启用并行计算
    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)

    # 在GPU上分配输出数组内存
    x_fft = gpuarray.empty_like(x_gpu, dtype=np.complex64)
    y_fft = gpuarray.empty_like(y_gpu, dtype=np.complex64)
    out_gpu = gpuarray.empty_like(x_gpu, dtype=np.complex64)

    # 执行FFT变换，将时域信号转换到频域
    fft.fft(x_gpu, x_fft, plan)
    fft.fft(y_gpu, y_fft, plan)

    # 在频域执行复数乘法，利用cuBLAS库的优化算法
    linalg.multiply(x_fft, y_fft, overwrite=True)

    # 执行IFFT变换，将频域结果转换回时域，并应用缩放因子
    fft.ifft(y_fft, out_gpu, inverse_plan, scale=True)
    conv_out = out_gpu.get()

    return conv_out


def conv_2d(ker, img):
    """
    执行2D卷积运算，使用零填充和循环移位技术

    核心思想：
    1. 零填充：扩展卷积核和图像尺寸，避免边界效应
    2. 循环移位：将卷积核中心对齐到填充后的数组中心
    3. FFT卷积：利用快速傅里叶变换加速卷积计算

    Args:
        ker: 卷积核（滤波器）
        img: 输入图像

    Returns:
        卷积后的图像
    """
    # 创建填充后的卷积核数组，尺寸为图像尺寸+2*卷积核尺寸
    # 这样设计确保卷积核能够完全覆盖图像的每个像素
    paddled_ker = np.zeros(
        (img.shape[0]+2*ker.shape[0], img.shape[1]+2*ker.shape[1])).astype(np.float32)
    paddled_ker[:ker.shape[0], :ker.shape[1]] = ker

    # paddled_ker 的四个角分别是高斯核被分割后的部分，每个部分近似为 15 * 15 大小（忽略中心元素）
    # 使用np.roll进行循环移位，将卷积核中心对齐到填充数组的中心位置
    # 这是实现"循环卷积"的关键步骤，确保卷积运算的几何正确性
    paddled_ker = np.roll(paddled_ker, shift=-ker.shape[0]//2, axis=0)
    paddled_ker = np.roll(paddled_ker, shift=-ker.shape[1]//2, axis=1)

    # 创建填充后的图像数组，中间放置原始图像
    paddled_img = np.zeros_like(paddled_ker).astype(np.float32)
    # paddled_img 中间放置了要处理的图像，图像距离四条边的距离都是 31
    # 这种填充策略确保卷积核能够处理图像边界，避免边界效应
    paddled_img[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]] = img

    # 调用GPU加速的FFT卷积函数
    out_ = cufft_conv(paddled_ker, paddled_img)

    # 提取有效输出区域，去除填充边界
    output = out_[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]]

    return output


def gaussian_filter(x, y, sigma):
    """
    计算2D高斯滤波器的单个像素值

    高斯函数：G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
    这是图像处理中最常用的平滑滤波器，具有各向同性和可分离性

    Args:
        x, y: 相对于滤波器中心的坐标
        sigma: 高斯函数的标准差，控制滤波器的平滑程度

    Returns:
        该位置的高斯权重值
    """
    return (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2*(sigma**2)))


def gaussian_ker(sigma):
    """
    生成2D高斯卷积核

    算法特点：
    1. 核尺寸为(2σ+1)×(2σ+1)，覆盖±σ范围内的所有像素
    2. 权重归一化，确保滤波后图像亮度不变
    3. 高斯核具有旋转对称性，适合各向同性的图像平滑

    Args:
        sigma: 高斯函数的标准差，值越大滤波效果越平滑

    Returns:
        归一化的高斯卷积核
    """
    ker_ = np.zeros((2*sigma+1, 2*sigma+1))
    for i in range(2*sigma+1):
        for j in range(2*sigma+1):
            ker_[i, j] = gaussian_filter(i-sigma, j-sigma, sigma)
    total_ = np.sum(ker_.ravel())

    # 归一化权重，确保滤波后图像的总亮度保持不变
    ker_ = ker_/total_

    return ker_


if __name__ == "__main__":
    # 主程序：演示高斯滤波在图像处理中的应用
    # 读取图像并归一化到[0,1]范围
    rei = np.float32(plt.imread("rei.jpg"))/255
    rei_blurred = np.zeros_like(rei)

    # 生成σ=15的高斯滤波器，核尺寸为31×31
    ker = gaussian_ker(15)

    # 对RGB三个通道分别执行卷积运算
    # 这种逐通道处理方式保持了图像的颜色信息
    for k in range(3):
        rei_blurred[:, :, k] = conv_2d(ker, rei[:, :, k])

    # 可视化结果：对比原始图像和滤波后的图像
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle("Guassian Filtering", fontsize=20)
    ax0.set_title("Before")
    ax0.axis("off")
    ax0.imshow(rei)
    ax1.set_title("After")
    ax1.axis("off")
    ax1.imshow(rei_blurred)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
