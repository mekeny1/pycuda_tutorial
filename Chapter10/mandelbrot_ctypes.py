'''
Author: mekeny1
Date: 2025-07-09 23:49:15
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:26:58
FilePath: \pycuda_tutorial_hapril\Chapter10\mandelbrot_ctypes.py
Description: 
Mandelbrot集合计算模块 - 使用ctypes调用CUDA加速的DLL
@overview: 通过ctypes接口调用预编译的CUDA DLL进行Mandelbrot集合计算
@architecture: Python + ctypes + CUDA DLL的混合架构，实现高性能分形计算
@algorithm: 基于复数迭代的Mandelbrot集合生成算法，GPU并行加速
@visualization: 集成matplotlib进行分形图像的可视化展示
@performance: 利用GPU并行计算能力显著提升分形图像生成速度
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from ctypes import *

# ==================== CUDA DLL 接口加载 ====================

# 加载预编译的CUDA Mandelbrot计算DLL
# 动态加载包含CUDA内核的Windows动态链接库
# 通过DLL接口调用GPU加速的Mandelbrot计算函数
mandel_dll = CDLL("./mandelbrot.dll")
mandel_c = mandel_dll.launch_mandelbrot

# 设置C函数参数类型和返回值类型
# 定义C函数接口的参数类型映射
# 确保Python数组与C数组的内存布局兼容
mandel_c.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_float, c_int]
# 参数类型：输入复数网格指针, 输出结果指针, 最大迭代次数, 上界阈值, 网格大小


def mandelbrot(breadth, low, high, max_iters, upper_bound):
    """
    Mandelbrot集合计算主函数

    @param breadth: 计算网格的宽度/高度（正方形网格）
    @param low: 复数平面的下界
    @param high: 复数平面的上界
    @param max_iters: 最大迭代次数，控制计算精度
    @param upper_bound: 逃逸半径阈值，判断点是否属于Mandelbrot集合
    @return: 包含每个点迭代次数的2D数组
    """

    # 创建复数平面的线性网格
    # 在指定范围内生成均匀分布的复数点
    # 使用numpy的高效数组操作
    lattice = np.linspace(low, high, breadth, dtype=np.float32)

    # 预分配输出结果数组
    # 避免动态内存分配，提升性能
    # 存储每个复数点的迭代次数
    out = np.empty(shape=(lattice.size, lattice.size), dtype=np.float32)

    # 调用CUDA加速的Mandelbrot计算函数
    # 将计算任务分发到GPU进行并行处理
    # 通过ctypes将Python数组转换为C指针
    # 利用GPU的数千个核心同时计算不同点的迭代
    mandel_c(lattice.ctypes.data_as(POINTER(c_float)), out.ctypes.data_as(
        POINTER(c_float)), c_int(max_iters), c_float(upper_bound), c_int(lattice.size))

    return out


if __name__ == "__main__":
    # ==================== 性能测试和可视化 ====================

    # 记录计算开始时间
    # 测量GPU加速计算的执行时间
    t1 = time()

    # 执行Mandelbrot集合计算
    # 生成512x512分辨率的Mandelbrot集合图像
    # 设置复数平面范围[-2,2]，最大迭代256次
    mandel = mandelbrot(512, -2, 2, 256, 2)

    # 记录计算结束时间
    t2 = time()
    mandel_time = t2-t1

    # 输出性能统计信息
    # 显示GPU加速相对于CPU计算的性能提升
    print('It took %s seconds to calculate the Mandelbrot graph.' % mandel_time)

    # 使用matplotlib可视化Mandelbrot集合
    # 将数值结果转换为彩色分形图像
    # 将迭代次数映射为颜色强度，创建经典的分形图案
    plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))  # 设置坐标轴范围
    plt.show()
