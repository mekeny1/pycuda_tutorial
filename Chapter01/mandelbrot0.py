'''
Author: mekeny1
Date: 2025-05-16 00:22:34
LastEditors: mekeny1
LastEditTime: 2025-08-11 00:58:40
FilePath: \pycuda_tutorial_hapril\Chapter01\mandelbrot0.py
Description: Mandelbrot集合CPU计算版本
result:
It took 6.075246810913086 seconds to calculate the Mandelbrot graph.
It took 0.21550917625427246 seconds to dump the image.
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from matplotlib import pyplot as plt
import numpy as np
from time import time
import matplotlib

# 设置matplotlib后端为agg，避免GUI依赖，适合服务器环境
matplotlib.use("agg")

"""
Mandelbrot集合计算程序 - CPU版本

该程序实现了Mandelbrot集合的经典算法，用于生成分形图像。
Mandelbrot集合是复平面上所有使得迭代序列 z_{n+1} = z_n^2 + c 不发散的复数c的集合。

算法核心思想：
1. 对于复平面上的每个点c，从z=0开始迭代
2. 计算z = z^2 + c，直到|z|超过阈值或达到最大迭代次数
3. 如果迭代过程中|z|始终小于阈值，则该点属于Mandelbrot集合

与CUDA的关系：
- 这是一个CPU串行版本，为后续GPU并行化提供基准
- 每个像素的计算相互独立，非常适合GPU并行处理
- 在GPU版本中，每个线程处理一个像素点，大幅提升计算效率

软硬件特性：
- CPU串行执行：当前版本使用CPU单线程计算，适合理解算法逻辑
- 内存访问模式：顺序访问数组，CPU缓存友好
- 浮点运算：使用complex64复数类型，平衡精度和性能
- 图像生成：使用matplotlib进行可视化，生成PNG格式图像
"""


def simple_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    """
    计算Mandelbrot集合

    算法流程：
    1. 在复平面上创建网格点
    2. 对每个点进行Mandelbrot迭代
    3. 根据迭代结果确定点是否属于集合

    参数:
        width (int): 图像宽度（像素数）
        height (int): 图像高度（像素数）
        real_low (float): 实轴下界
        real_high (float): 实轴上界
        imag_low (float): 虚轴下界
        imag_high (float): 虚轴上界
        max_iters (int): 最大迭代次数
        upper_bound (float): 发散阈值

    返回:
        numpy.ndarray: 形状为(height, width)的二维数组，1表示属于集合，0表示不属于
    """
    # 在复平面上创建均匀分布的网格点
    # real_vals: 实轴上的值，对应x坐标
    real_vals = np.linspace(real_low, real_high, width)
    # image_vals: 虚轴上的值，对应y坐标
    image_vals = np.linspace(imag_low, imag_high, height)

    # 初始化结果数组，默认所有点都属于集合（值为1）
    mandelbrot_graph = np.ones((height, width), dtype=np.float32)

    # 双重循环遍历每个像素点
    # 外层循环：x坐标（实轴）
    for x in range(width):
        # 内层循环：y坐标（虚轴）
        for y in range(height):
            # 构造复数c = a + bi，其中a是实部，b是虚部
            # 这里使用complex64类型，与GPU计算保持一致
            c = np.complex64(real_vals[x] + image_vals[y] * 1j)
            # 初始化迭代变量z = 0
            z = np.complex64(0)

            # Mandelbrot迭代：z = z^2 + c
            for i in range(max_iters):
                # 核心迭代公式：z_{n+1} = z_n^2 + c
                z = z**2 + c

                # 检查是否发散：如果|z| > upper_bound，则点不属于集合
                if (np.abs(z) > upper_bound):
                    # 标记该点不属于Mandelbrot集合
                    mandelbrot_graph[y, x] = 0
                    break
                # 如果迭代完成仍未发散，则该点属于集合（保持默认值1）

    return mandelbrot_graph


if __name__ == "__main__":
    # 记录计算开始时间
    t1 = time()

    # 计算Mandelbrot集合
    # 参数说明：
    # - 512x512: 图像分辨率，适合GPU并行处理
    # - (-2, 2, -2, 2): 复平面范围，覆盖Mandelbrot集合的主要区域
    # - 256: 最大迭代次数，影响图像细节和计算精度
    # - 2: 发散阈值，通常设为2
    mandel = simple_mandelbrot(512, 512, -2, 2, -2, 2, 256, 2)

    # 记录计算结束时间
    t2 = time()
    mandel_time = t2 - t1

    # 记录图像保存开始时间
    t1 = time()

    # 创建图像并保存
    fig = plt.figure(1)
    # 显示Mandelbrot集合，extent参数指定坐标轴范围
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    # 保存为PNG格式，dpi参数控制图像分辨率
    plt.savefig("mandelbrot.png", dpi=fig.dpi)

    # 记录图像保存结束时间
    t2 = time()
    dump_time = t2 - t1

    # 输出性能统计信息
    print("计算Mandelbrot图像耗时: {} 秒".format(mandel_time))
    print("保存图像耗时: {} 秒".format(dump_time))
