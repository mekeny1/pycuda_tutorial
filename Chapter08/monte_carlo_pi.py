'''
Author: mekeny1
Date: 2025-06-28 14:49:18
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:18:36
FilePath: \pycuda_tutorial_hapril\Chapter08\monte_carlo_pi.py
Description: Monte Carlo Pi计算器 - 基于CUDA的圆周率估算实现
    本模块使用Monte Carlo方法通过GPU并行计算来估算圆周率π的值。
    算法基于几何概率：在单位正方形内随机生成点，计算落在单位圆内的
    点的比例，从而估算π/4的值，最终得到π的近似值。

Core Features:
    - GPU并行Monte Carlo Pi估算算法
    - 基于几何概率的圆周率计算
    - 高精度随机数生成
    - 符号计算精度验证
    - 与NumPy常量的对比验证

Algorithm:
    - 在[0,1]×[0,1]单位正方形内随机生成点(x,y)
    - 计算点到原点的距离：sqrt(x²+y²)
    - 统计距离≤1的点数（落在单位圆内）
    - 通过比例关系：π/4 = 圆内点数/总点数
    - 最终得到：π = 4 × (圆内点数/总点数)

Hardware Requirements:
    - NVIDIA GPU with CUDA support
    - CUDA Toolkit installed
    - PyCUDA library

Software Dependencies:
    - pycuda.autoinit: CUDA上下文初始化
    - pycuda.driver: CUDA驱动接口
    - pycuda.gpuarray: GPU数组操作
    - pycuda.compiler: CUDA内核编译
    - numpy: 数值计算支持
    - sympy: 符号计算支持
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from sympy import Rational

# CUDA内核代码 - 实现Monte Carlo Pi估算的核心算法
ker = SourceModule(no_extern_c=True, source="""
    #include<curand_kernel.h>
    #define _PYTHAG(a, b) (a * a + b * b)  // 计算点到原点距离的平方（避免开方运算）
    #define ULL unsigned long long

    extern "C"
    {
        // Monte Carlo Pi估算内核函数
        // 每个线程负责生成指定数量的随机点并统计落在圆内的点数
        __global__ void estimate_pi(ULL iters, ULL *hits)
        {
            curandState cr_state;  // CUDA随机数生成器状态
            int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程ID

            // 初始化随机数生成器 - 使用时钟和线程ID确保随机性
            curand_init((ULL)clock() + (ULL)tid, (ULL)0, (ULL)0, &cr_state);

            float x, y;  // 随机点的坐标

            // Monte Carlo采样循环 - 每个线程生成iters个随机点
            for (ULL i = 0; i < iters; i++)
            {
                // 生成[0,1)区间内的均匀随机坐标
                x = curand_uniform(&cr_state);
                y = curand_uniform(&cr_state);

                // 检查点是否落在单位圆内（距离原点≤1）
                // 使用距离平方比较避免开方运算，提高性能
                if (_PYTHAG(x, y) <= 1.0f)
                {
                    hits[tid]++;  // 统计圆内点数
                }
            }

            return;
        }
    }
""")

# 获取编译后的CUDA内核函数引用
pi_ker = ker.get_function("estimate_pi")

# GPU线程配置参数
threads_per_block = 32    # 每个线程块的线程数
blocks_per_grid = 512     # 网格中的线程块数
total_threads = threads_per_block*blocks_per_grid  # 总线程数

# 在GPU上分配结果存储数组 - 每个线程存储其统计的圆内点数
hits_d = gpuarray.zeros((total_threads,), dtype=np.uint64)

# 每个线程的采样次数 - 使用2^24确保足够的精度
iters = 2**24

# 启动CUDA内核 - 执行并行Monte Carlo Pi计算
# 参数：采样次数, 结果数组, 网格配置, 线程块配置
pi_ker(np.uint64(iters), hits_d, grid=(
    blocks_per_grid, 1, 1), block=(threads_per_block, 1, 1))

# 将GPU结果传输到CPU并计算总和
total_hits = np.sum(hits_d.get())  # 所有线程统计的圆内点总数
total = np.uint64(total_threads)*np.uint64(iters)  # 总采样点数

# 使用符号计算进行精确的π估算
# π = 4 × (圆内点数/总点数)
est_pi_symbolic = Rational(4)*Rational(int(total_hits), int(total))

# 将符号结果转换为浮点数
est_pi = np.float32(est_pi_symbolic.evalf())

# 输出计算结果和验证
print("Our Monte Carlo estimate of Pi is: %s" % est_pi)
print("NumPy's Pi constant is: %s" % np.pi)
print("Our estimate passes NumPy's 'allclose': %s" %
      np.allclose(est_pi, np.pi))
