'''
Author: mekeny1
Date: 2025-07-10 10:06:19
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:28:19
FilePath: \pycuda_tutorial_hapril\Chapter10\mandelbrot_ptx.py
Description: 
Mandelbrot集合计算模块 - 使用PyCUDA和PTX内核
@overview: 通过PyCUDA高级接口调用PTX内核进行Mandelbrot集合计算
@architecture: Python + PyCUDA + PTX内核的简化GPU编程架构
@algorithm: 基于复数迭代的Mandelbrot集合生成算法，GPU并行加速
@memory_management: 使用PyCUDA的gpuarray自动管理GPU内存
@usability: 相比Driver API更简洁的接口，适合快速GPU编程开发
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pycuda
from pycuda import gpuarray
import pycuda.autoinit
import pycuda.driver

# ==================== PTX模块加载 ====================

# 从PTX文件加载CUDA模块
# 使用PyCUDA的高级接口直接从PTX文件加载编译好的CUDA内核
mandel_mod = pycuda.driver.module_from_file("./mandelbrot.ptx")

# 获取内核函数引用
# 从加载的模块中获取名为"mandelbrot_ker"的内核函数
mandel_ker = mandel_mod.get_function("mandelbrot_ker")


def mandelbrot(breadth, low, high, max_iters, upper_bound):
    """
    Mandelbrot集合计算主函数 - 使用PyCUDA和PTX内核

    参数:
        breadth: 计算网格的宽度/高度（正方形网格）
        low: 复数平面的下界
        high: 复数平面的上界
        max_iters: 最大迭代次数，控制计算精度
        upper_bound: 逃逸半径阈值，判断点是否属于Mandelbrot集合
    返回:
        包含每个点迭代次数的2D数组
    """

    # 创建复数平面的线性网格并传输到GPU
    # 使用numpy生成均匀分布的复数点，然后通过gpuarray自动传输到GPU
    lattice = gpuarray.to_gpu(np.linspace(
        low, high, breadth, dtype=np.float32))

    # 在GPU上预分配输出结果数组
    # 使用gpuarray.empty创建GPU内存数组，存储每个点的迭代次数
    out_gpu = gpuarray.empty(
        shape=(lattice.size, lattice.size), dtype=np.float32)

    # 计算线程网格配置
    # 根据数据大小计算最优的线程网格布局，确保每个GPU核心都有足够的工作负载
    gridsize = int(np.ceil(lattice.size**2/32))

    # 启动CUDA内核函数
    # 使用PyCUDA的简化接口启动内核，自动处理参数类型转换
    # grid和block参数配置线程网格和线程块结构
    mandel_ker(lattice, out_gpu, np.int32(256), np.float32(upper_bound**2),
               np.int32(lattice.size), grid=(gridsize, 1, 1), block=(32, 1, 1))

    # 将计算结果从GPU传输回CPU
    # 使用gpuarray.get()方法自动将GPU数据复制到CPU内存
    out = out_gpu.get()

    return out


if __name__ == "__main__":
    # ==================== 性能测试和可视化 ====================

    # 记录计算开始时间
    # 测量PyCUDA + PTX方法的执行时间
    t1 = time()

    # 执行Mandelbrot集合计算
    # 生成512x512分辨率的Mandelbrot集合图像
    # 设置复数平面范围[-2,2]，最大迭代256次
    mandel = mandelbrot(512, -2, 2, 256, 2)

    # 计算并输出性能统计信息
    # 显示PyCUDA + PTX方法相对于其他方法的性能
    mandel_time = time()-t1
    print("It took %s seconds to calculate the Mandelbrot graph." % mandel_time)

    # 使用matplotlib可视化Mandelbrot集合
    # 将数值结果转换为彩色分形图像
    # 将迭代次数映射为颜色强度，创建经典的分形图案
    plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.show()
