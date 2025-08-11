'''
Author: mekeny1
Date: 2025-06-15 01:37:06
LastEditors: mekeny1
LastEditTime: 2025-08-11 16:02:08
FilePath: \pycuda_tutorial_hapril\Chapter06\01.hello-world_gpu.py
Description: 使用 PyCUDA 编译并启动一个简单的 CUDA 内核，在 GPU 上打印线程与线程块信息（Hello World 示例）
#cuda: CUDA内核编程入门与线程管理基础
#parallel: GPU并行计算模型与线程层次结构
#debug: 设备端printf调试技术与输出管理
#hardware: GPU硬件架构与线程调度机制
#pycuda: PyCUDA框架使用与CUDA内核编译
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
# PyCUDA自动初始化模块，自动创建CUDA上下文并绑定默认GPU设备
# 完成必要的GPU初始化工作，包括内存管理器和CUDA运行时环境设置
import pycuda.autoinit
# CUDA源码编译器模块，提供JIT（即时编译）功能
# 将CUDA C代码编译为可执行的GPU内核函数
from pycuda.compiler import SourceModule

"""
代码总览（Chinese Overview）
- 目标：演示如何通过 PyCUDA 在 GPU 上编译/加载并发射一个最小化的 CUDA 内核，使用设备端 printf 输出线程与线程块信息。
- 核心步骤：
  1) 使用 pycuda.autoinit 自动创建 CUDA 上下文（绑定默认 GPU 并完成必要初始化）。
  2) 通过 SourceModule 对 CUDA C 源码进行 JIT 编译并加载至当前上下文，得到可调用的内核函数。
  3) 以给定的 grid/block 维度发射内核；设备端 printf 的输出会在内核完成后刷新到主机 stdout。
- CUDA/硬件要点：
  - 并行层级：grid 由多个 block 组成，block 由多个 thread 组成；在内核中可通过 threadIdx、blockIdx、blockDim、gridDim 获取坐标与规模。
  - 设备端 printf 仅用于调试，会有额外的性能开销，且多线程打印的先后次序不保证。
  - GPU 内核发射通常是异步的；本示例规模很小，脚本结束或上下文销毁前会完成必要同步并刷新输出。
- 本例配置：grid=(2,1,1)，block=(5,1,1) → 共 2×5=10 个线程，每个线程打印一行标识。
"""

# 使用SourceModule编译CUDA源码，创建可执行的GPU内核模块
# 这是PyCUDA的核心功能，将CUDA C代码转换为GPU可执行的机器码
ker = SourceModule(
    """
    // 全局内核函数：GPU并行执行的入口点
    // __global__ 修饰符表示这是一个从主机端调用的GPU内核函数
    // 每个线程都会执行这个函数的副本，通过内置变量区分不同线程
    __global__ void hello_world_ker()
    {
        // 设备端printf：每个线程输出自己的标识信息
        // threadIdx.x：当前线程在线程块内的索引（0到blockDim.x-1）
        // blockIdx.x：当前线程块在网格内的索引（0到gridDim.x-1）
        // 注意：多线程printf的输出顺序不保证，主要用于调试目的
        
        printf("Hello world from thread %d, in block %d!\\n",threadIdx.x,blockIdx.x);
        
        // 条件执行：只有第一个线程块中的第一个线程执行此代码段
        // 这演示了如何在线程间进行条件分支控制
        // 注意：过多的条件分支会导致线程分歧，影响GPU性能
        if(threadIdx.x==0 && blockIdx.x==0)
        {
            // 输出分隔线，便于识别特殊输出
            printf("-------------------------------------\\n");
            // 输出网格配置信息：总共有多少个线程块
            // gridDim.x：网格在x维度的线程块数量
            printf("This kernel was launched over a grid consisting of %d blocks,\\n", gridDim.x);
            // 输出线程块配置信息：每个线程块有多少个线程
            // blockDim.x：线程块在x维度的线程数量
            printf("where each block has %d threads.\\n", blockDim.x);
        }
    }

"""
)

# 从编译好的CUDA模块中获取内核函数的Python调用句柄
# 这个句柄用于在Python代码中调用GPU内核函数
# get_function()方法返回一个可调用的Python函数对象
hello_ker = ker.get_function("hello_world_ker")

# 发射GPU内核执行：配置线程层次结构并启动并行计算
# 参数说明：
# - block=(5,1,1)：每个线程块包含5×1×1=5个线程
# - grid=(2,1,1)：网格包含2×1×1=2个线程块
# - 总计：2×5=10个线程并行执行
#
# CUDA线程层次结构：
# - Grid（网格）：最高层级的线程组织，包含多个线程块
# - Block（线程块）：中间层级的线程组织，包含多个线程
# - Thread（线程）：最低层级的执行单元，每个线程执行内核函数的一个副本
#
# 硬件执行机制：
# - GPU将线程组织为warp（通常是32个线程）
# - 同一warp内的线程以SIMT（单指令多线程）方式执行
# - 不同warp可以并行执行，提高GPU利用率
#
# 注意：设备端printf输出顺序依赖GPU调度和缓冲机制，主要用于调试
hello_ker(block=(5, 1, 1), grid=(2, 1, 1))
