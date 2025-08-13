'''
Author: mekeny1
Date: 2025-07-15 19:50:17
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:29:42
FilePath: \pycuda_tutorial_hapril\Chapter11\shfl_xor.py
Description: CUDA shuffle xor指令演示，展示线程束内基于XOR模式的数据交换和通信模式
Tags: cuda, warp-shuffle, shfl-xor, thread-communication, data-exchange, gpu-computing, parallel-algorithms
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# CUDA C 代码，演示 __shfl_xor_sync 的用法，实现线程束内数据交换
# shuffle xor指令是warp shuffle的一种特殊模式，基于XOR操作实现线程间数据交换
ShflCode = """
__global__ void shfl_xor_ker(int *input, int *output)
{
    int temp = input[threadIdx.x]; // 每个线程读取一个输入元素
    // 每个线程获取自己的输入数据，准备进行XOR模式的数据交换

    // cuda11不支持__shfl_xor, 使用__shfl_xor_sync代替
    // 这里 mask=所有线程，laneMask=1，width=blockDim.x
    // __shfl_xor_sync实现基于XOR的数据交换：线程i与线程(i^laneMask)交换数据
    // 当laneMask=1时，相邻线程对之间进行数据交换（0↔1, 2↔3, 4↔5, ...）
    temp = __shfl_xor_sync(0xFFFFFFFF, temp, 1, blockDim.x);

    output[threadIdx.x] = temp; // 写回结果
    // 每个线程将交换后的数据写回输出数组
}
"""

# 编译 CUDA 代码，获取核函数
# 这个核函数使用了现代CUDA的同步shuffle指令，需要CUDA 11+支持
shfl_mod = SourceModule(ShflCode)
shfl_ker = shfl_mod.get_function("shfl_xor_ker")

# 构造输入数据（0~31）并拷贝到 GPU
# 使用32个连续整数作为测试数据，便于观察XOR交换的效果
dinput = gpuarray.to_gpu(np.int32(np.int32(range(32))))
doutput = gpuarray.empty_like(dinput)

# 启动核函数，block 大小为 32
# block大小为32确保每个block正好是一个warp，最大化shuffle指令效率
shfl_ker(dinput, doutput, grid=(1, 1, 1), block=(32, 1, 1))

print("input array: %s" % dinput.get())
print("array after __shfl_xor_sync: %s" % doutput.get())
# 输出结果将显示相邻线程对之间的数据交换效果
