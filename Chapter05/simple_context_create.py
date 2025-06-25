import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv

# 初始化cuda
drv.init()

# 选择gpu设备
dev=drv.Device(0)

ctx=dev.make_context()

x=gpuarray.to_gpu(np.float32([1,2,3]))
print(x.get())

# 销毁cuda上下文
ctx.pop()
