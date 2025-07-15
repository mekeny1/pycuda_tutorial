import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv

# CUDA C 代码，定义了演示原子操作的核函数
AtomicCode = """
// 核函数，演示原子加法、原子最大值、原子交换
__global__ void atomic_ker(int *add_out,int *max_out)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x; // 计算全局线程索引

    // 原子交换：将 *add_out 设置为 0，所有线程安全地执行
    atomicExch(add_out,0);
    __syncthreads(); // 线程同步，确保所有线程都已完成上一步

    // 原子加法：每个线程对 *add_out 加 1
    atomicAdd(add_out,1);

    // 原子最大值：所有线程将自己的 tid 与 max_out 比较，写入最大值
    atomicMax(max_out,tid);
}
"""

# 编译 CUDA 代码，获取核函数
atomic_mod = SourceModule(AtomicCode)
atomic_ker = atomic_mod.get_function("atomic_ker")

# 分配 GPU 输出内存
add_out = gpuarray.empty((1,), dtype=np.int32)
max_out = gpuarray.empty((1,), dtype=np.int32)

# 启动核函数，block 里有 100 个线程
atomic_ker(add_out, max_out, grid=(1, 1, 1), block=(100, 1, 1))
# 同步，确保 GPU 计算完成
drv.Context.synchronize()

print("Atomic operations test: ")
# 打印原子加法结果（应为线程数 100）
print("add_out: %s" % add_out.get()[0])
# 打印原子最大值结果（应为最大线程 id 99）
print("max_out: %s" % max_out.get()[0])
