import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# CUDA C 代码，演示 __shfl_xor_sync 的用法，实现线程束内数据交换
ShflCode = """
__global__ void shfl_xor_ker(int *input, int *output)
{
    int temp = input[threadIdx.x]; // 每个线程读取一个输入元素

    // cuda11不支持__shfl_xor, 使用__shfl_xor_sync代替
    // 这里 mask=所有线程，laneMask=1，width=blockDim.x
    temp = __shfl_xor_sync(0xFFFFFFFF, temp, 1, blockDim.x);

    output[threadIdx.x] = temp; // 写回结果
}
"""

# 编译 CUDA 代码，获取核函数
shfl_mod = SourceModule(ShflCode)
shfl_ker = shfl_mod.get_function("shfl_xor_ker")

# 构造输入数据（0~31）并拷贝到 GPU
dinput = gpuarray.to_gpu(np.int32(np.int32(range(32))))
doutput = gpuarray.empty_like(dinput)

# 启动核函数，block 大小为 32
shfl_ker(dinput, doutput, grid=(1, 1, 1), block=(32, 1, 1))

print("input array: %s" % dinput.get())
print("array after __shfl_xor_sync: %s" % doutput.get())
