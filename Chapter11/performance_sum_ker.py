'''
Author: mekeny1
Date: 2025-07-15 23:05:11
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:29:26
FilePath: \pycuda_tutorial_hapril\Chapter11\performance_sum_ker.py
Description: CUDA高性能求和核函数，展示warp shuffle指令、向量化内存访问和性能优化技术
Tags: cuda, performance-optimization, warp-shuffle, atomic-operations, vectorized-memory, gpu-computing, reduction-algorithm
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from timeit import timeit

# CUDA C 代码，定义了用于高性能求和的核函数和辅助函数
# 这个实现使用了多种GPU性能优化技术：warp shuffle、向量化内存访问、原子操作
SumCode = """
// 获取当前线程的 lane id（线程束内编号）
// 使用内联汇编直接访问硬件寄存器，获取线程在warp中的位置
__device__ int __inline__ laneid()
{
    int id;
    asm("mov.u32 %0,%%laneid;":"=r"(id));
    return id;
}

// 将 double 拆分为两个 int（低32位和高32位）
// 这是为了配合warp shuffle指令，因为shuffle只支持32位整数
__device__ void __inline__ split64(double val,int *lo,int *hi)
{
    asm volatile("mov.b64 {%0,%1},%2;":"=r"(*lo),"=r"(*hi):"d"(val));
}

// 将两个 int（低32位和高32位）合成为 double
// 与split64配合使用，实现double类型在warp shuffle中的传输
__device__ void __inline__ combine64(double *val,int lo,int hi)
{
    asm volatile("mov.b64 %0,{%1,%2};":"=d"(*val):"r"(lo),"r"(hi));
}

// 主核函数：对输入数组进行高性能归约求和
// 使用warp shuffle指令实现高效的线程间通信，避免共享内存访问
__global__ void sum_ker(double *input,double *out)
{
    int id = laneid(); // 获取线程束内编号

    // 每个线程处理两个 double 元素
    // 使用double2向量化内存访问，提高内存带宽利用率
    double2 vals = *reinterpret_cast<double2*>(&input[(blockDim.x*blockIdx.x+threadIdx.x)*2]);

    double sum_val = vals.x + vals.y; // 先对本线程的两个元素求和
    double temp = 0.0;

    int s1 = 0, s2 = 0;

    // warp 内归约，利用 shuffle 指令高效通信
    // 这是GPU上最高效的归约算法，避免了共享内存的bank冲突
    for (int i = 1; i < 32; i *= 2)
    {
        split64(sum_val, &s1, &s2); // 拆分 double 为两个 int

        // 线程束内下移归约
        // __shfl_down_sync是Volta架构引入的同步shuffle指令
        // 0xffffffff表示所有32个线程都参与同步
        s1 = __shfl_down_sync(0xffffffff, s1, i, 32);
        s2 = __shfl_down_sync(0xffffffff, s2, i, 32);

        combine64(&temp, s1, s2); // 合并回 double
        sum_val += temp; // 累加
    }

    // 线程束内第一个线程将结果原子加到输出
    // 使用原子操作确保多个warp的结果正确累加
    if(id == 0)
    {
        atomicAdd(out, sum_val);
    }
}
"""

# 编译 CUDA 代码，获取核函数
# 这个核函数使用了高级CUDA特性，需要现代GPU架构支持
sum_mod = SourceModule(SumCode)
sum_ker = sum_mod.get_function("sum_ker")

# 生成测试数据（长度为 10000*2*32 的 double 数组）
# 数据量足够大以体现性能差异，且是32的倍数以充分利用warp
a = np.float64(np.random.randn(10000*2*32))
a_gpu = gpuarray.to_gpu(a)  # 拷贝到 GPU
out = gpuarray.zeros((1,), dtype=np.float64)  # 输出初始化为 0

# 启动核函数，grid 大小按数据量分配，block 大小为 32
# block大小为32确保每个block正好是一个warp，最大化shuffle指令效率
sum_ker(a_gpu, out, grid=(int(np.ceil(a.size/64)), 1, 1), block=(32, 1, 1))
drv.Context.synchronize()  # 等待 GPU 计算完成

# 校验 GPU 求和结果与 NumPy 求和是否一致
# 使用allclose进行浮点数比较，考虑数值精度误差
print("Does sum_ker produces the same value as NumPy's sum (according allclose)? : %s" %
      np.allclose(np.sum(a), out.get()[0]))

print("Performing sum_ker / PyCUDA sum timing tests (20 each)...")

# 计时：自定义核函数求和
# 测试20次取平均值，获得稳定的性能数据
sum_ker_time = timeit(
    """from __main__ import sum_ker,a_gpu,out,np,drv \nsum_ker(a_gpu,out,grid=(int(np.ceil(a_gpu.size/64)),1,1),block=(32,1,1)) \ndrv.Context.synchronize()""", number=20)

# 计时：PyCUDA 自带的 gpuarray.sum 求和
# 对比PyCUDA内置的求和函数性能
pycuda_sum_time = timeit(
    """from __main__ import gpuarray, a_gpu, drv \ngpuarray.sum(a_gpu) \ndrv.Context.synchronize()""", number=20)

# 输出两种方法的平均耗时和性能提升比
# 展示自定义优化核函数相对于通用库的性能优势
print("sum_ker average time duration: %s, PyCUDA's gpuarray.sum average time duration: %s" %
      (sum_ker_time, pycuda_sum_time))
print("(Performance improvement of sum_ker over gpuarray.sum: %s )" %
      (pycuda_sum_time/sum_ker_time))
