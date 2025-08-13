'''
Author: mekeny1
Date: 2025-07-10 10:47:40
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:28:14
FilePath: \pycuda_tutorial_hapril\Chapter10\mandelbrot_driver.py
Description: 
Mandelbrot集合计算模块 - 使用CUDA Driver API直接编程
@overview: 通过CUDA Driver API实现底层GPU编程，直接控制CUDA内核执行
@architecture: Python + CUDA Driver API + PTX内核的纯GPU计算架构
@algorithm: 基于复数迭代的Mandelbrot集合生成算法，完全GPU并行化
@memory_management: 手动管理GPU内存分配、数据传输和释放
@performance: 直接使用Driver API实现最优性能，避免高级库的开销
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from cuda_driver import *


def mandelbrot(breadth, low, high, max_iters, upper_bound):
    """
    Mandelbrot集合计算主函数 - 使用CUDA Driver API

    @param breadth: 计算网格的宽度/高度（正方形网格）
    @param low: 复数平面的下界
    @param high: 复数平面的上界
    @param max_iters: 最大迭代次数，控制计算精度
    @param upper_bound: 逃逸半径阈值，判断点是否属于Mandelbrot集合
    @return: 包含每个点迭代次数的2D数组
    """

    # ==================== CUDA Driver API 初始化 ====================

    # 初始化CUDA Driver API
    # 启动CUDA Driver API，准备GPU操作
    cuInit(0)

    # 获取系统中可用的GPU设备数量
    # 检查系统中是否有可用的NVIDIA GPU
    cnt = c_int(0)
    cuDeviceGetCount(byref(cnt))

    # 检查GPU设备可用性
    # 如果没有找到GPU设备，抛出异常
    if cnt.value == 0:
        raise Exception("No GPU device found!")

    # 获取第一个GPU设备句柄
    # 选择索引为0的GPU设备进行操作
    cuDevice = c_int(0)
    cuDeviceGet(byref(cuDevice), 0)

    # 创建CUDA上下文
    # 为选定的GPU设备创建执行上下文
    # 上下文管理GPU资源和内存
    cuContext = c_void_p()
    cuCtxCreate(byref(cuContext), 0, cuDevice)

    # ==================== PTX模块加载 ====================

    # 加载包含CUDA内核的PTX模块文件
    # 从文件系统加载编译好的PTX代码
    # 使PTX中的内核函数可用于执行
    cuModule = c_void_p()
    # 需要添加TypeError: bytes or integer address expected instead of str instance
    cuModuleLoad(byref(cuModule), c_char_p(b"./mandelbrot.ptx"))

    # ==================== 内存分配和数据准备 ====================

    # 创建复数平面的线性网格
    # 在指定范围内生成均匀分布的复数点
    # 使用numpy的高效数组操作
    lattice = np.linspace(low, high, breadth, dtype=np.float32)
    lattice_c = lattice.ctypes.data_as(POINTER(c_float))

    # 在GPU上分配输入数据内存
    # 为复数网格数据分配GPU全局内存
    # 分配lattice.size个float32元素的内存空间
    lattice_gpu = c_void_p(0)
    graph = np.zeros(shape=(lattice.size, lattice.size), dtype=np.float32)
    cuMemAlloc(byref(lattice_gpu), c_size_t(lattice.size*sizeof(c_float)))

    # 在GPU上分配输出结果内存
    # 为计算结果分配GPU内存，存储每个点的迭代次数
    # 使用2D数组布局存储计算结果
    graph_gpu = c_void_p(0)
    cuMemAlloc(byref(graph_gpu), c_size_t(lattice.size**2*sizeof(c_float)))

    # 将输入数据从CPU复制到GPU
    # 数据传输方向：CPU内存 → GPU内存
    # 异步内存传输，启动GPU计算前的数据准备
    cuMemcpyHtoD(lattice_gpu, lattice_c, c_size_t(
        lattice.size*sizeof(c_float)))

    # ==================== 内核函数准备 ====================

    # 从PTX模块中获取内核函数句柄
    # 根据函数名获取CUDA内核函数句柄
    # 为内核启动准备函数引用
    mandel_ker = c_void_p(0)
    cuModuleGetFunction(byref(mandel_ker), cuModule,
                        c_char_p(b"mandelbrot_ker"))

    # 准备内核函数参数
    # 将Python参数转换为C类型
    # 构建内核函数所需的参数结构
    max_iters = c_int(max_iters)
    upper_bound_squared = c_float(upper_bound**2)  # 预计算平方值避免重复计算
    lattice_size = c_int(lattice.size)

    # 构建内核函数参数数组
    # 创建内核函数参数的内存布局
    # 将参数转换为指针数组供内核使用
    mandel_args0 = [lattice_gpu, graph_gpu,
                    max_iters, upper_bound_squared, lattice_size]
    mandel_args = [c_void_p(addressof(x)) for x in mandel_args0]
    mandel_params = (c_void_p * len(mandel_args))(*mandel_args)

    # ==================== 内核执行 ====================

    # 计算线程网格配置
    # 根据数据大小计算最优的线程网格布局
    # 确保每个GPU核心都有足够的工作负载
    gridsize = int(np.ceil(lattice.size**2/32))

    # 启动CUDA内核函数
    # 在GPU上执行并行Mandelbrot计算
    # 配置线程块大小(32,1,1)和网格大小
    # 为每个线程块分配10000字节共享内存
    cuLaunchKernel(mandel_ker, gridsize, 1, 1, 32, 1,
                   1, 10000, None, mandel_params, None)

    # 同步GPU操作
    # 等待所有GPU计算完成
    # 确保计算结果已写入GPU内存
    cuCtxSynchronize()

    # ==================== 结果获取和清理 ====================

    # 将计算结果从GPU复制到CPU
    # 数据传输方向：GPU内存 → CPU内存
    # 获取GPU计算的分形图像数据
    cuMemcpyDtoH(cast(graph.ctypes.data, c_void_p), graph_gpu,
                 c_size_t(lattice.size**2*sizeof(c_float)))

    # 释放GPU内存资源
    # 释放之前分配的GPU内存，防止内存泄漏
    # 确保GPU资源正确释放
    cuMemFree(lattice_gpu)
    cuMemFree(graph_gpu)

    # 销毁CUDA上下文
    # 释放CUDA上下文占用的所有资源
    cuCtxDestroy(cuContext)

    return graph


if __name__ == "__main__":
    # ==================== 性能测试和可视化 ====================

    # 记录计算开始时间
    # 测量CUDA Driver API计算的执行时间
    t1 = time()

    # 执行Mandelbrot集合计算
    # 生成512x512分辨率的Mandelbrot集合图像
    # 设置复数平面范围[-2,2]，最大迭代256次
    mandel = mandelbrot(512, -2, 2, 256, 2)

    # 计算并输出性能统计信息
    # 显示CUDA Driver API相对于其他方法的性能
    mandel_time = time()-t1
    print("It took %s seconds to calculate the Mandelbrot graph." % mandel_time)

    # 使用matplotlib可视化Mandelbrot集合
    # 将数值结果转换为彩色分形图像
    # 将迭代次数映射为颜色强度，创建经典的分形图案
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.show()
