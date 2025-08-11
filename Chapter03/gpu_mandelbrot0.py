'''
Author: mekeny1
Date: 2025-05-25 17:01:59
LastEditors: mekeny1
LastEditTime: 2025-08-11 01:27:54
FilePath: \pycuda_tutorial_hapril\Chapter03\gpu_mandelbrot0.py
Description: GPU并行化Mandelbrot集合计算程序 - 使用PyCUDA ElementwiseKernel实现
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from pycuda.elementwise import ElementwiseKernel
from pycuda import gpuarray
import pycuda.autoinit
import numpy as np
from matplotlib import pyplot as plt
from time import time
import matplotlib

# 设置matplotlib后端为agg，避免GUI依赖，适合服务器环境
matplotlib.use("agg")

"""
GPU并行化Mandelbrot集合计算程序

该程序使用PyCUDA的ElementwiseKernel实现Mandelbrot集合的GPU并行计算。
相比CPU版本，GPU版本能够同时处理多个像素点，大幅提升计算效率。

算法处理流程：
1. 在CPU上创建复平面网格（lattice）
2. 将网格数据传输到GPU内存
3. 在GPU上并行执行Mandelbrot迭代计算
4. 将计算结果从GPU传输回CPU
5. 生成可视化图像

核心方法：
- ElementwiseKernel: PyCUDA提供的元素级并行计算内核
- gpuarray.to_gpu(): 将CPU数据转移到GPU内存
- gpuarray.empty(): 在GPU上分配内存空间
- kernel.get(): 将GPU计算结果传输回CPU

CUDA相关概念：
- 内核（Kernel）：在GPU上并行执行的函数
- 元素级并行：每个线程处理一个数组元素
- 全局内存：GPU上可被所有线程访问的内存
- 内存传输：CPU和GPU之间的数据传输开销

软硬件特性：
- GPU并行架构：数千个CUDA核心同时执行相同指令
- 内存层次：GPU全局内存、共享内存、寄存器
- SIMT执行模型：单指令多线程，自动线程调度
- 内存带宽：GPU内存带宽远高于CPU，适合大规模并行计算
"""

# 定义GPU内核函数 - 使用CUDA C++语法
# 每个线程处理一个复数点，执行Mandelbrot迭代
mandel_ker = ElementwiseKernel(
    # 参数列表：输入网格、输出结果、最大迭代次数、发散阈值
    "pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
    """
    // 默认该点属于Mandelbrot集合
    mandelbrot_graph[i]=1;
    // 获取当前线程对应的复数点c
    pycuda::complex<float> c=lattice[i];
    // 初始化迭代变量z = 0
    pycuda::complex<float> z(0,0);
    // Mandelbrot迭代循环
    for (int j=0; j<max_iters; j++) 
    {
        // 核心迭代公式：z = z^2 + c
        z=z*z+c;
        // 检查是否发散
        if (abs(z) > upper_bound) 
        {
            // 如果发散，标记该点不属于集合
            mandelbrot_graph[i]=0;
            break;
        }
    }
    """,
    "mandelbrot_ker"  # 内核函数名称
)


def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    """
    GPU并行计算Mandelbrot集合

    算法流程：
    1. 创建复平面网格（CPU端）
    2. 数据传输到GPU
    3. 并行执行Mandelbrot计算
    4. 结果传回CPU

    参数:
        width (int): 图像宽度
        height (int): 图像高度
        real_low/high (float): 实轴范围
        imag_low/high (float): 虚轴范围
        max_iters (int): 最大迭代次数
        upper_bound (float): 发散阈值

    返回:
        numpy.ndarray: Mandelbrot集合图像数据
    """
    # 创建实轴上的值（复数形式）
    real_vals = np.matrix(np.linspace(
        real_low, real_high, width), dtype=np.complex64)
    # 创建虚轴上的值（复数形式）
    image_vals = np.matrix(np.linspace(
        imag_low, imag_high, height), dtype=np.complex64)*1j

    # 构造复平面网格：每个点对应一个复数c = a + bi
    # 使用矩阵运算创建完整的复平面网格
    mandelbrot_lattice = np.array(
        real_vals+image_vals.transpose(), dtype=np.complex64)

    # 将复平面网格数据传输到GPU内存
    # 这是GPU计算的第一步：数据准备
    mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)

    # 在GPU上分配结果数组内存
    # 每个元素存储对应点是否属于Mandelbrot集合（1或0）
    mandelbrot_graph_gpu = gpuarray.empty(
        shape=mandelbrot_lattice.shape, dtype=np.float32)

    # 在GPU上并行执行Mandelbrot计算
    # 每个线程处理一个像素点，同时进行迭代计算
    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu,
               np.int32(max_iters), np.float32(upper_bound))

    # 将计算结果从GPU传输回CPU内存
    # 这是GPU计算的最后一步：结果获取
    mandelbrot_graph = mandelbrot_graph_gpu.get()

    return mandelbrot_graph


if __name__ == "__main__":
    # 记录GPU计算开始时间
    t1 = time()

    # 执行GPU并行Mandelbrot计算
    # 参数与CPU版本相同，但计算方式完全不同
    mandel = gpu_mandelbrot(512, 512, -2, 2, -2, 2, 256, 2)

    # 记录GPU计算结束时间
    t2 = time()
    mandel_time = t2 - t1

    # 记录图像保存开始时间
    t1 = time()

    # 创建并保存Mandelbrot图像
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)

    # 记录图像保存结束时间
    t2 = time()
    dump_time = t2 - t1

    # 输出性能统计信息
    # 与CPU版本对比，GPU版本应该有显著的性能提升
    print('GPU计算Mandelbrot图像耗时: {} 秒'.format(mandel_time))
    print('保存图像耗时: {} 秒'.format(dump_time))
