'''
Author: mekeny1
Date: 2025-05-31 00:18:53
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:37:44
FilePath: \pycuda_tutorial_hapril\Chapter04\conway_gpu.py
Description: 使用PyCUDA实现康威生命游戏动画版本，采用双缓冲技术避免数据竞争，支持实时可视化细胞演化过程
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as dvr
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 康威生命游戏GPU实现 - 动画版本（双缓冲技术）
# =============================================================================
# 算法核心思想：
# 1. 使用双缓冲技术：lattice_gpu存储当前状态，newlattice_gpu存储下一状态
# 2. 避免数据竞争：读取当前状态计算下一状态，避免读写冲突
# 3. 支持实时动画：每帧更新一次，实现细胞演化的可视化
#
# 硬件特性利用：
# - GPU并行计算：支持更大网格(128x128)，4096个线程块并行处理
# - 内存管理：双缓冲避免内存访问冲突，提高计算效率
# - 实时渲染：GPU计算与CPU显示分离，实现流畅动画效果
# =============================================================================

ker = SourceModule(
    """
    #define _X (threadIdx.x + blockIdx.x * blockDim.x)
    #define _Y (threadIdx.y + blockIdx.y * blockDim.y)

    #define _WIDTH (blockDim.x * gridDim.x)
    #define _HEIGHT (blockDim.y * gridDim.y)

    #define _XM(x) ((x + _WIDTH) % _WIDTH)
    #define _YM(y) ((y + _HEIGHT) % _HEIGHT)

    #define _INDEX(x, y) (_XM(x) + _YM(y) * _WIDTH)

    // 计算指定细胞周围8个邻居中存活细胞的数量
    // 使用环形边界条件处理边缘细胞
    __device__ int nbrs(int x, int y, int *in)
    {
        return (
            in[_INDEX(x - 1, y + 1)] + in[_INDEX(x - 1, y)] + in[_INDEX(x - 1, y - 1)] +
            in[_INDEX(x, y + 1)] + in[_INDEX(x, y - 1)] +
            in[_INDEX(x + 1, y + 1)] + in[_INDEX(x + 1, y)] + in[_INDEX(x + 1, y - 1)]);
    }

    // 康威生命游戏核心内核函数
    // 使用双缓冲技术：读取lattice，写入lattice_out
    __global__ void conway_ker(int *lattice_out, int *lattice)
    {
        int x = _X, y = _Y;

        int n = nbrs(x, y, lattice);  // 计算邻居数量

        // 康威生命游戏规则实现
        if(lattice[_INDEX(x,y)]==1)  // 当前细胞存活
        {
            switch (n)
            {
                case 2:  // 2个邻居：保持存活
                case 3:  // 3个邻居：保持存活
                    lattice_out[_INDEX(x,y)]=1;
                    break;
                default:  // 其他情况：死亡
                    lattice_out[_INDEX(x,y)]=0;
            }
        }
        else if(lattice[_INDEX(x,y)]==0)  // 当前细胞死亡
        {
            switch (n)
            {
                case 3:  // 3个邻居：复活
                    lattice_out[_INDEX(x,y)]=1;
                    break;
                default:  // 其他情况：保持死亡
                    lattice_out[_INDEX(x,y)]=0;
            }
        }
    }
    """
)


conway_ker = ker.get_function("conway_ker")


def update_gpu(frameNum, img, newLattice_gpu, lattice_gpu, N):
    # 使用双缓冲技术更新GPU状态：读取lattice_gpu，写入newLattice_gpu
    # 网格维度使用整除确保为整数：N//32 x N//32
    conway_ker(newLattice_gpu, lattice_gpu, grid=(
        N//32, N//32, 1), block=(32, 32, 1))

    # 更新matplotlib图像显示
    img.set_data(newLattice_gpu.get())

    # 交换缓冲区：将新状态复制到当前状态
    lattice_gpu[:] = newLattice_gpu[:]

    return img


if __name__ == "__main__":
    # 设置网格大小 - 128x128，支持更大规模模拟
    N = 128

    # 初始化随机细胞状态，25%概率存活，75%概率死亡
    lattice = np.int32(np.random.choice(
        [1, 0], N*N, p=[0.25, 0.75]).reshape(N, N))
    lattice_gpu = gpuarray.to_gpu(lattice)

    # 创建输出缓冲区，避免内存访问冲突
    newlattice_gpu = gpuarray.empty_like(lattice_gpu)

    # 设置matplotlib动画：每帧调用update_gpu函数
    fig, ax = plt.subplots()
    img = ax.imshow(lattice_gpu.get(), interpolation="nearest")
    ani = animation.FuncAnimation(fig, update_gpu, fargs=(
        img, newlattice_gpu, lattice_gpu, N,), interval=0, frames=1000, save_count=1000)

    plt.show()
