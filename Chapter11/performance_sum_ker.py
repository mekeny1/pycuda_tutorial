import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from timeit import timeit

# CUDA C 代码，定义了用于高性能求和的核函数和辅助函数
SumCode = """
// 获取当前线程的 lane id（线程束内编号）
__device__ int __inline__ laneid()
{
    int id;
    asm("mov.u32 %0,%%laneid;":"=r"(id));
    return id;
}

// 将 double 拆分为两个 int（低32位和高32位）
__device__ void __inline__ split64(double val,int *lo,int *hi)
{
    asm volatile("mov.b64 {%0,%1},%2;":"=r"(*lo),"=r"(*hi):"d"(val));
}

// 将两个 int（低32位和高32位）合成为 double
__device__ void __inline__ combine64(double *val,int lo,int hi)
{
    asm volatile("mov.b64 %0,{%1,%2};":"=d"(*val):"r"(lo),"r"(hi));
}

// 主核函数：对输入数组进行高性能归约求和
__global__ void sum_ker(double *input,double *out)
{
    int id = laneid(); // 获取线程束内编号

    // 每个线程处理两个 double 元素
    double2 vals = *reinterpret_cast<double2*>(&input[(blockDim.x*blockIdx.x+threadIdx.x)*2]);

    double sum_val = vals.x + vals.y; // 先对本线程的两个元素求和
    double temp = 0.0;

    int s1 = 0, s2 = 0;

    // warp 内归约，利用 shuffle 指令高效通信
    for (int i = 1; i < 32; i *= 2)
    {
        split64(sum_val, &s1, &s2); // 拆分 double 为两个 int

        // 线程束内下移归约
        s1 = __shfl_down_sync(0xffffffff, s1, i, 32);
        s2 = __shfl_down_sync(0xffffffff, s2, i, 32);

        combine64(&temp, s1, s2); // 合并回 double
        sum_val += temp; // 累加
    }

    // 线程束内第一个线程将结果原子加到输出
    if(id == 0)
    {
        atomicAdd(out, sum_val);
    }
}
"""

# 编译 CUDA 代码，获取核函数
sum_mod = SourceModule(SumCode)
sum_ker = sum_mod.get_function("sum_ker")

# 生成测试数据（长度为 10000*2*32 的 double 数组）
a = np.float64(np.random.randn(10000*2*32))
a_gpu = gpuarray.to_gpu(a)  # 拷贝到 GPU
out = gpuarray.zeros((1,), dtype=np.float64)  # 输出初始化为 0

# 启动核函数，grid 大小按数据量分配，block 大小为 32
sum_ker(a_gpu, out, grid=(int(np.ceil(a.size/64)), 1, 1), block=(32, 1, 1))
drv.Context.synchronize()  # 等待 GPU 计算完成

# 校验 GPU 求和结果与 NumPy 求和是否一致
print("Does sum_ker produces the same value as NumPy's sum (according allclose)? : %s" %
      np.allclose(np.sum(a), out.get()[0]))

print("Performing sum_ker / PyCUDA sum timing tests (20 each)...")

# 计时：自定义核函数求和
sum_ker_time = timeit(
    """from __main__ import sum_ker,a_gpu,out,np,drv \nsum_ker(a_gpu,out,grid=(int(np.ceil(a_gpu.size/64)),1,1),block=(32,1,1)) \ndrv.Context.synchronize()""", number=20)

# 计时：PyCUDA 自带的 gpuarray.sum 求和
pycuda_sum_time = timeit(
    """from __main__ import gpuarray, a_gpu, drv \ngpuarray.sum(a_gpu) \ndrv.Context.synchronize()""", number=20)

# 输出两种方法的平均耗时和性能提升比
print("sum_ker average time duration: %s, PyCUDA's gpuarray.sum average time duration: %s" %
      (sum_ker_time, pycuda_sum_time))
print("(Performance improvement of sum_ker over gpuarray.sum: %s )" %
      (pycuda_sum_time/sum_ker_time))
