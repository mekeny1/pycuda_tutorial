import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# CUDA C 代码，演示使用 __shfl_down_sync 进行线程束内归约求和
ShflSumCode = """
__global__ void shfl_sum_ker(int *input, int *output)
{
    int temp = input[threadIdx.x]; // 每个线程读取一个输入元素

    // 线程束内归约求和，利用 shuffle 指令高效通信
    for (int i = 1; i < 32; i *= 2)
    {
        // mask 控制哪些线程参与 shuffle 操作。一般用 0xFFFFFFFF 表示全部线程参与，特殊情况下可以用自定义的 mask 或 __activemask() 以保证正确性和安全性。
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i, 32);
    }

    // 线程束内第一个线程写出最终结果
    if (threadIdx.x == 0)
    {
        *output = temp;
    }
}
"""

# 编译 CUDA 代码，获取核函数
shfl_mod = SourceModule(ShflSumCode)
shfl_sum_ker = shfl_mod.get_function("shfl_sum_ker")

# 构造输入数据（0~31）并拷贝到 GPU
array_in = gpuarray.to_gpu(np.int32(range(32)))
output = gpuarray.empty((1,), dtype=np.int32)

# 启动核函数，block 大小为 32
shfl_sum_ker(array_in, output, grid=(1, 1, 1), block=(32, 1, 1))

print("Input array: %s" % array_in.get())
print("Summed value: %s" % output.get()[0])
# 校验 GPU 归约结果与 Python sum 是否一致
print("Does this match with python's sum? : %s" %
      (output.get()[0] == sum(array_in.get())))
