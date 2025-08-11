'''
Author: mekeny1
Date: 2025-06-02 15:41:23
LastEditors: mekeny1
LastEditTime: 2025-08-11 10:20:55
FilePath: \pycuda_tutorial_hapril\Chapter04\conway_gpu_syncthreads_shared.py
Description: 使用PyCUDA实现康威生命游戏，采用共享内存优化和同步线程机制，展示GPU并行计算在细胞自动机模拟中的应用
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as dvr
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
from time import time

# =============================================================================
# 康威生命游戏GPU实现 - 共享内存优化版本
# =============================================================================
# 算法核心思想：
# 1. 使用共享内存(Shared Memory)减少全局内存访问延迟
# 2. 通过__syncthreads()确保线程间数据同步
# 3. 利用GPU的SIMT(单指令多线程)架构并行处理所有细胞状态
#
# 硬件特性利用：
# - GPU共享内存：每个Block内的线程共享32KB快速内存，访问延迟极低
# - 线程块同步：__syncthreads()确保Block内所有线程到达同步点后再继续执行
# - 内存合并访问：相邻线程访问相邻内存地址，提高内存带宽利用率
# =============================================================================

shared_ker = SourceModule(
    """
    #define _iters 1000000

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
    // 使用共享内存优化，减少全局内存访问次数
    __global__ void conway_ker_shared(int *p_lattice,int iters) // p_lattice位于全局内存中
    {
        int x = _X, y = _Y;
        __shared__ int lattice[32*32];  // 声明32x32的共享内存数组

        // 将全局内存数据复制到共享内存
        lattice[_INDEX(x,y)]=p_lattice[_INDEX(x,y)];
        __syncthreads(); // 确保所有线程完成数据复制后再继续执行
                                                
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
            lattice[_INDEX(x,y)]=cell_value;  // 更新共享内存中的细胞状态
            __syncthreads();  // 同步点2：确保所有线程完成状态更新
        }

        // 最终同步：确保所有迭代完成
        __syncthreads();
        p_lattice[_INDEX(x,y)]=lattice[_INDEX(x,y)];  // 将结果写回全局内存
        __syncthreads();  // 最终同步：确保所有线程完成数据写回
    }
    """
)


conway_ker_shared = shared_ker.get_function("conway_ker_shared")


if __name__ == "__main__":
    # 设置网格大小 - 32x32适合单个Block处理
    N = 32

    # 初始化随机细胞状态，25%概率存活，75%概率死亡
    lattice = np.int32(np.random.choice(
        [1, 0], N*N, p=[0.25, 0.75]).reshape(N, N))
    lattice_gpu = gpuarray.to_gpu(lattice)

    # 执行GPU内核：1个Grid，32x32的Block，100万次迭代
    conway_ker_shared(lattice_gpu, np.int32(1000000),
                      grid=(1, 1, 1), block=(32, 32, 1))

    # 可视化最终结果
    fig = plt.figure(1)
    plt.imshow(lattice_gpu.get())
    plt.show()
