'''
Author: mekeny1
Date: 2025-06-13 01:16:43
LastEditors: mekeny1
LastEditTime: 2025-08-11 11:13:35
FilePath: \pycuda_tutorial_hapril\Chapter05\gpu_mandelbrot_context_sync.py
Description: 使用PyCUDA实现曼德博集合(Mandelbrot Set)的GPU并行计算版本，演示CUDA上下文同步机制和异步内存操作
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda import gpuarray
import numpy as np
from matplotlib import pyplot as plt
from time import time
import matplotlib
matplotlib.use("agg")

"""
代码总体说明：
本程序实现了曼德博集合的GPU并行计算版本，主要特点包括：

1. 算法思想：
   - 曼德博集合是复平面上满足特定条件的复数集合
   - 对于复数c，计算迭代序列 z(n+1) = z(n)^2 + c，z(0) = 0
   - 如果序列发散（|z| > 上界），则c不在集合中；否则c在集合中

2. GPU并行化策略：
   - 使用ElementwiseKernel实现逐元素并行计算
   - 每个GPU线程负责计算一个复数点的曼德博集合判断
   - 利用GPU的大量并行线程同时处理所有复数点

3. CUDA上下文同步机制：
   - 使用context.synchronize()确保GPU操作完成
   - 在关键步骤后同步，保证数据一致性
   - 演示了异步操作与同步操作的配合使用

4. 软硬件特性利用：
   - GPU：并行计算复数迭代，每个线程独立处理一个点
   - CPU：负责数据准备、结果可视化和文件保存
   - 内存：使用GPU全局内存存储复数网格和计算结果
   - 异步传输：减少CPU-GPU等待时间，提高整体性能
"""

# 定义ElementwiseKernel，实现曼德博集合的核心计算逻辑
# 每个线程处理一个复数点，计算其是否属于曼德博集合
mandel_ker = ElementwiseKernel(
    "pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
    """
    mandelbrot_graph[i]=1;  // 初始假设该点属于曼德博集合
    pycuda::complex<float> c=lattice[i];  // 获取当前复数点
    pycuda::complex<float> z(0,0);  // 初始化迭代变量z为0
    
    // 执行曼德博集合的迭代计算
    for (int j=0; j<max_iters; j++) 
    {
        z=z*z+c;  // 核心迭代公式：z = z^2 + c
        
        // 检查是否发散（超出上界）
        if (abs(z) > upper_bound) 
        {
            mandelbrot_graph[i]=0;  // 发散，标记为不属于集合
            break;  // 提前退出循环
        }
    }
    // 如果循环完成仍未发散，则保持mandelbrot_graph[i]=1，表示属于集合
    """,
    "mandelbrot_ker" // CUDA核函数名称
)


def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    """
    在GPU上计算曼德博集合

    参数说明：
    - width, height: 输出图像的宽度和高度
    - real_low, real_high: 实轴的范围
    - imag_low, imag_high: 虚轴的范围
    - max_iters: 最大迭代次数
    - upper_bound: 发散判断的上界值

    返回：
    - mandelbrot_graph: 曼德博集合的二值图像（1表示属于集合，0表示不属于）
    """
    # 创建复数网格：实部和虚部的笛卡尔积
    real_vals = np.matrix(np.linspace(
        real_low, real_high, width), dtype=np.complex64)  # 实轴上的值
    image_vals = np.matrix(np.linspace(
        imag_low, imag_high, height), dtype=np.complex64)*1j  # 虚轴上的值（乘以1j）

    # 将实部和虚部组合成复数网格
    mandelbrot_lattice = np.array(
        real_vals+image_vals.transpose(), dtype=np.complex64)

    # 异步复制复数网格到GPU内存
    mandelbrot_lattice_gpu = gpuarray.to_gpu_async(mandelbrot_lattice)
    # 同步等待数据传输完成，确保数据完整性
    pycuda.autoinit.context.synchronize()

    # 在GPU上为计算结果分配内存空间
    mandelbrot_graph_gpu = gpuarray.empty(
        shape=mandelbrot_lattice.shape, dtype=np.float32)

    # 在GPU上执行曼德博集合计算核函数
    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu,
               np.int32(max_iters), np.float32(upper_bound))
    # 同步等待计算完成，确保结果可用
    pycuda.autoinit.context.synchronize()

    # 异步获取计算结果
    mandelbrot_graph = mandelbrot_graph_gpu.get_async()
    # 同步等待数据传输完成，确保数据完整性
    pycuda.autoinit.context.synchronize()

    return mandelbrot_graph


if __name__ == "__main__":
    # 记录计算开始时间
    t1 = time()

    # 计算512x512分辨率的曼德博集合
    # 范围：实轴[-2,2]，虚轴[-2,2]，最大迭代256次，发散上界2.0
    mandel = gpu_mandelbrot(512, 512, -2, 2, -2, 2, 256, 2)

    # 记录计算结束时间
    t2 = time()

    # 计算GPU计算耗时
    mandel_time = t2 - t1

    # 记录图像保存开始时间
    t1 = time()

    # 创建matplotlib图形并显示曼德博集合
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))  # 设置坐标轴范围
    plt.savefig('mandelbrot.png', dpi=fig.dpi)  # 保存为PNG文件

    # 记录图像保存结束时间
    t2 = time()

    # 计算图像保存耗时
    dump_time = t2 - t1

    # 输出性能统计信息
    print('It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time))
    print('It took {} seconds to dump the image.'.format(dump_time))
