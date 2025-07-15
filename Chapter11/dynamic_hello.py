import numpy as np
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit

# CUDA C 代码，演示动态并行和递归 kernel 调用
DynamicParallelsmCode = """
// 动态并行核函数，递归调用自身
__global__ void dynamic_hello_ker(int depth)
{
    // 每个线程打印自己的编号和递归深度
    printf("Hello from thread %d, recursion depth %d!\\n", threadIdx.x, depth);
    // 仅在第一个线程、第一块、block 大小大于 1 时递归 launch 新 kernel
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockDim.x > 1)
    {
        printf("Launching a new kernel from depth %d .\\n",depth);
        printf("-----------------------------------------\\n");
        // 动态 launch 一个新的 kernel，block 数不变，blockDim.x-1
        dynamic_hello_ker<<<1, blockDim.x - 1>>>(depth + 1);
    }
}
"""

# 编译动态并行 CUDA 代码，获取核函数
dp_mod = DynamicSourceModule(DynamicParallelsmCode)
hello_ker = dp_mod.get_function("dynamic_hello_ker")
# 启动 kernel，初始递归深度为 0，block 大小为 4
hello_ker(np.int32(0), grid=(1, 1, 1), block=(4, 1, 1))
