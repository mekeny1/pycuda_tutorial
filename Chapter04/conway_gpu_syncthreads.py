'''
Author: mekeny1
Date: 2025-06-02 12:44:20
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:36:42
FilePath: \pycuda_tutorial_hapril\Chapter04\conway_gpu_syncthreads.py
Description: 使用PyCUDA实现康威生命游戏基础版本，通过线程同步机制确保数据一致性，展示GPU并行计算在细胞自动机模拟中的基本应用
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as dvr
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 康威生命游戏GPU实现 - 基础版本（无共享内存优化）
# =============================================================================
# 算法核心思想：
# 1. 直接操作全局内存，每个线程独立处理一个细胞
# 2. 使用__syncthreads()确保所有线程完成状态更新后再进行下一轮计算
# 3. 利用GPU的SIMT架构并行执行生命游戏规则
#
# 硬件特性利用：
# - GPU全局内存：所有线程可访问的统一内存空间
# - 线程同步：__syncthreads()确保Block内线程执行顺序一致性
# - 并行计算：1024个线程(32x32)同时计算细胞状态
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
    // 直接操作全局内存，通过同步确保数据一致性
    __global__ void conway_ker(int *lattice,int iters)
    {
        int x = _X, y = _Y;
                                                
        // 主循环：执行指定次数的生命游戏规则
        for(int i=0;i<iters;i++)
        {
            int n = nbrs(x, y, lattice);  // 计算邻居数量
            int cell_value;

            // 康威生命游戏规则实现
            if(lattice[_INDEX(x,y)]==1)  // 当前细胞存活
            {
                switch (n)
                {
                    case 2:  // 2个邻居：保持存活
                    case 3:  // 3个邻居：保持存活
                        cell_value=1;
                        break;
                    default:  // 其他情况：死亡
                        cell_value=0;
                }
            }
            else if(lattice[_INDEX(x,y)]==0)  // 当前细胞死亡
            {
                switch (n)
                {
                    case 3:  // 3个邻居：复活
                        cell_value=1;
                        break;
                    default:  // 其他情况：保持死亡
                        cell_value=0;
                }
            }
            
            // 同步点1：确保所有线程完成状态计算
            __syncthreads();
            lattice[_INDEX(x,y)]=cell_value;  // 更新全局内存中的细胞状态
            __syncthreads();  // 同步点2：确保所有线程完成状态更新
        }
    }
    """
)


conway_ker = ker.get_function("conway_ker")


if __name__ == "__main__":
    # 设置网格大小 - 32x32适合单个Block处理
    N = 32

    # 初始化随机细胞状态，25%概率存活，75%概率死亡
    lattice = np.int32(np.random.choice(
        [1, 0], N*N, p=[0.25, 0.75]).reshape(N, N))
    lattice_gpu = gpuarray.to_gpu(lattice)

    # 执行GPU内核：1个Grid，32x32的Block，100万次迭代
    conway_ker(lattice_gpu, np.int32(1000000),
               grid=(1, 1, 1), block=(32, 32, 1))

    # 可视化最终结果
    fig = plt.figure(1)
    plt.imshow(lattice_gpu.get())

    plt.show()
