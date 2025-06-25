import numpy as np
# 自动初始化 CUDA 环境
import pycuda.autoinit
# 提供 GPU 上的数组操作
from pycuda import gpuarray
# 用于创建 GPU 上的前缀和计算内核
from pycuda.scan import InclusiveScanKernel

seq=np.array([1,2,3,4],dtype=np.int32)
seq_gpu=gpuarray.to_gpu(seq)
# 包含性前缀和（Inclusive Scan）：每个输出位置包含当前元素及之前所有元素的和
sum_gpu=InclusiveScanKernel(np.int32,"a+b")
print(sum_gpu(seq_gpu).get())
print(np.cumsum(seq))

# [ 1  3  6 10]
# [ 1  3  6 10]