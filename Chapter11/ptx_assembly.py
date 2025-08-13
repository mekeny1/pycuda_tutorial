'''
Author: mekeny1
Date: 2025-07-15 21:09:25
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:29:32
FilePath: \pycuda_tutorial_hapril\Chapter11\ptx_assembly.py
Description: CUDA PTX内联汇编演示程序，展示GPU底层指令集和汇编级编程技术
Tags: cuda, ptx-assembly, inline-assembly, gpu-instructions, low-level-programming, hardware-registers, gpu-computing
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# CUDA C 代码，演示 PTX 内联汇编的各种用法
# PTX（Parallel Thread Execution）是NVIDIA GPU的中间表示语言，允许直接访问底层硬件
PtxCode = """
extern "C" {
    // 将 int 变量置零
    // 使用mov.s32指令将寄存器置零，s32表示有符号32位整数
    __device__ void set_to_zero(int *x)
    {
        asm("mov.s32 %0, 0;" : "=r"(*x));
    }

    // 执行 float 加法，结果写入 out
    // 使用add.f32指令进行单精度浮点加法，f32表示32位浮点数
    __device__ void add_floats(float *out, float in1, float in2)
    {
        asm("add.f32 %0, %1, %2;" : "=f"(*out) : "f"(in1), "f"(in2));
    }

    // int 自增
    // 使用add.s32指令实现自增操作，+r表示读写寄存器
    __device__ void plusplus(int *x)
    {
        asm("add.s32 %0, %0, 1;" : "+r"(*x));
    }

    // 获取当前线程的 lane id
    // 直接访问硬件寄存器%%laneid获取线程在warp中的位置
    __device__ int laneid()
    {
        int id;
        asm("mov.u32 %0, %%laneid;" : "=r"(id));
        return id;
    }

    // 将 double 拆分为两个 int（低32位和高32位）
    // 使用mov.b64指令处理64位数据，b64表示64位数据
    __device__ void split64(double val, int *lo, int *hi)
    {
        asm volatile("mov.b64 {%0,%1},%2;" : "=r"(*lo), "=r"(*hi) : "d"(val));
    }

    // 将两个 int 合成为 double
    // 与split64配合使用，实现64位数据的拆分和重组
    __device__ void combined64(double *val, int lo, int hi)
    {
        asm volatile("mov.b64 %0,{%1,%2};" : "=d"(*val) : "r"(lo), "r"(hi));
    }

    // 主核函数，演示各种 PTX 内联汇编用法
    // 这个函数展示了PTX汇编在GPU编程中的实际应用
    __global__ void ptx_test_ker()
    {
        int x = 123;
        printf("x is %d \\n", x);

        set_to_zero(&x); // 置零
        // 使用PTX汇编指令将变量置零，验证汇编函数的正确性
        printf("x is now %d \\n", x);

        plusplus(&x); // 自增
        // 使用PTX汇编指令实现自增操作
        printf("x is now %d \\n", x);

        float f = 0.0f;
        printf("f is %f \\n", f);

        add_floats(&f, 1.11f, 2.22f); // 浮点加法
        // 使用PTX汇编指令进行浮点运算，验证精度和正确性
        printf("f is now %f \\n", f);

        printf("lane ID is %d \\n", laneid()); // 打印 lane id
        // 获取并显示当前线程在warp中的位置

        double orig = 3.1415;
        int t1 = 0, t2 = 0;
        split64(orig, &t1, &t2); // double 拆分
        // 将64位double拆分为两个32位int，用于数据传输或存储

        double recon = 0.0;
        combined64(&recon, t1, t2); // 合成 double
        // 将两个32位int重新组合为64位double，验证拆分/组合的正确性
        printf("Do split64 / combined64 work? : %s \\n", (orig == recon) ? "true" : "false");
    }
}
"""

# 编译 CUDA 代码，获取核函数
# PTX汇编代码需要特殊的编译处理，确保汇编指令正确生成
ptx_mod = SourceModule(PtxCode)
ptx_test_ker = ptx_mod.get_function("ptx_test_ker")

# 启动核函数，单线程测试 PTX 汇编功能
# 使用单线程执行，便于观察PTX汇编指令的执行结果
ptx_test_ker(grid=(1, 1, 1), block=(1, 1, 1))
