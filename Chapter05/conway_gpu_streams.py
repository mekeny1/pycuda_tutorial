'''
Author: mekeny1
Date: 2025-06-12 00:44:20
LastEditors: mekeny1
LastEditTime: 2025-08-11 11:10:10
FilePath: \pycuda_tutorial_hapril\Chapter05\conway_gpu_streams.py
Description: 使用PyCUDA实现康威生命游戏(Conway's Game of Life)的GPU并行计算版本，支持多流并发处理以提高性能
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
代码总体说明：
本程序实现了康威生命游戏的GPU并行计算版本，主要特点包括：

1. 算法思想：
   - 康威生命游戏是一种细胞自动机，每个细胞的生死状态由其周围8个邻居细胞决定
   - 规则：活细胞周围有2-3个活邻居则存活，死细胞周围有3个活邻居则复活，否则死亡

2. GPU并行化策略：
   - 使用CUDA核函数在GPU上并行计算每个细胞的新状态
   - 采用多流(Streams)并发处理多个游戏实例，提高GPU利用率
   - 每个流处理一个独立的游戏网格，实现真正的并行计算

3. 软硬件特性利用：
   - GPU：利用大量并行线程同时计算所有细胞状态
   - CPU：负责数据准备、结果可视化等串行任务
   - 内存：使用GPU全局内存存储游戏网格，支持异步数据传输

4. 性能优化：
   - 网格维度设置为32x32的倍数，充分利用GPU的warp结构
   - 使用异步内存操作和流同步，减少CPU-GPU等待时间
   - 支持多个游戏实例同时运行，提高整体吞吐量
"""

# 定义CUDA核函数，实现康威生命游戏的核心计算逻辑
ker = SourceModule(
    """
    // 线程索引宏定义，用于计算当前线程在网格中的位置
    #define _X (threadIdx.x + blockIdx.x * blockDim.x)
    #define _Y (threadIdx.y + blockIdx.y * blockDim.y)

    // 网格尺寸宏定义，表示整个计算网格的宽度和高度
    #define _WIDTH (blockDim.x * gridDim.x)
    #define _HEIGHT (blockDim.y * gridDim.y)

    // 边界处理宏定义，实现周期性边界条件（环形网格）
    #define _XM(x) ((x + _WIDTH) % _WIDTH)
    #define _YM(y) ((y + _HEIGHT) % _HEIGHT)

    // 一维索引计算宏定义，将2D坐标转换为1D数组索引
    #define _INDEX(x, y) (_XM(x) + _YM(y) * _WIDTH)

    // 计算指定位置周围8个邻居中活细胞的数量
    // 这是康威生命游戏的核心计算，决定细胞的下一个状态
    __device__ int nbrs(int x, int y, int *in)
    {
        return (
            in[_INDEX(x - 1, y + 1)] + in[_INDEX(x - 1, y)] + in[_INDEX(x - 1, y - 1)] +
            in[_INDEX(x, y + 1)] + in[_INDEX(x, y - 1)] +
            in[_INDEX(x + 1, y + 1)] + in[_INDEX(x + 1, y)] + in[_INDEX(x + 1, y - 1)]);
    }

    // 主要的CUDA核函数，每个线程负责计算一个细胞的新状态
    // 使用switch语句实现康威生命游戏的规则逻辑
    __global__ void conway_ker(int *lattice_out, int *lattice)
    {
        int x = _X, y = _Y;  // 获取当前线程对应的网格位置

        int n = nbrs(x, y, lattice);  // 计算邻居中活细胞的数量

        if(lattice[_INDEX(x,y)]==1)  // 如果当前细胞是活的
        {
            switch (n)
            {
                case 2:  // 2个活邻居：保持存活
                case 3:  // 3个活邻居：保持存活
                    lattice_out[_INDEX(x,y)]=1;
                    break;
                default:  // 其他情况：死亡
                    lattice_out[_INDEX(x,y)]=0;
            }
        }
        else if(lattice[_INDEX(x,y)]==0)  // 如果当前细胞是死的
        {
            switch (n)
            {
                case 3:  // 3个活邻居：复活
                    lattice_out[_INDEX(x,y)]=1;
                    break;
                default:  // 其他情况：保持死亡
                    lattice_out[_INDEX(x,y)]=0;
            }
        }
    }
    """
)


# 从编译后的模块中获取CUDA核函数
conway_ker = ker.get_function("conway_ker")


def update_gpu(frameNum, imgs, newLattices_gpu, lattices_gpu, N, streams, num_concurrent):
    """
    更新GPU上的游戏状态，支持多流并发处理

    参数说明：
    - frameNum: 当前帧数（matplotlib动画回调使用）
    - imgs: matplotlib图像对象列表，用于显示每个游戏实例
    - newLattices_gpu: GPU上的新状态数组列表
    - lattices_gpu: GPU上的当前状态数组列表
    - N: 游戏网格的尺寸（N x N）
    - streams: CUDA流列表，用于并发执行
    - num_concurrent: 并发执行的游戏实例数量
    """
    for k in range(num_concurrent):
        # 在指定的CUDA流上执行康威生命游戏核函数
        # 网格维度设置为(N//32, N//32, 1)，确保每个块处理32x32的网格区域
        # 块维度设置为(32, 32, 1)，充分利用GPU的warp结构（32线程一组）
        conway_ker(newLattices_gpu[k], lattices_gpu[k], grid=(
            N//32, N//32, 1), block=(32, 32, 1), stream=streams[k])

        # 异步获取计算结果并更新显示图像
        imgs[k].set_data(newLattices_gpu[k].get_async(stream=streams[k]))

        # 将新状态异步复制回当前状态数组，为下一帧计算做准备
        lattices_gpu[k].set_async(newLattices_gpu[k], stream=streams[k])

    return imgs


if __name__ == "__main__":

    N = 128  # 游戏网格尺寸：128 x 128
    num_concurrent = 4  # 并发执行的游戏实例数量

    # 初始化CUDA流和GPU数组
    streams = []  # 存储多个CUDA流，用于并发执行
    lattices_gpu = []  # 存储多个游戏网格的当前状态
    newLattices_gpu = []  # 存储多个游戏网格的新状态

    for k in range(num_concurrent):
        # 为每个游戏实例创建一个独立的CUDA流
        streams.append(drv.Stream())

        # 随机初始化游戏网格，25%概率为活细胞，75%概率为死细胞
        lattice = np.int32(np.random.choice(
            [1, 0], N*N, p=[0.25, 0.75]).reshape(N, N))

        # 将初始状态传输到GPU内存
        lattices_gpu.append(gpuarray.to_gpu(lattice))

        # 在GPU上为新状态分配内存空间
        newLattices_gpu.append(gpuarray.empty_like(lattices_gpu[k]))

    # 创建matplotlib图形，显示多个并发的游戏实例
    fig, ax = plt.subplots(nrows=1, ncols=num_concurrent)
    imgs = []

    for k in range(num_concurrent):
        # 为每个游戏实例创建图像显示对象
        # 使用异步获取确保与对应的CUDA流同步
        imgs.append(ax[k].imshow(lattices_gpu[k].get_async(
            stream=streams[k]), interpolation="nearest"))

    # 创建动画，每帧调用update_gpu函数更新游戏状态
    # interval=0表示尽可能快地更新，frames=1000设置最大帧数
    ani = animation.FuncAnimation(fig, update_gpu, fargs=(
        imgs, newLattices_gpu, lattices_gpu, N, streams, num_concurrent), interval=0, frames=1000, save_count=1000)

    plt.show()
