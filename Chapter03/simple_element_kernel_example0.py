import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel

host_data = np.float32(np.random.random(50000000))
# 涉及python的命名空间
gpu_2x_ker=ElementwiseKernel(
    "float *in, float *out",
    "out[i] = in[i] * 2;",
    # 涉及CUDA C的命名空间
    "gpu_2x_ker"
)


def speedcomparison():
    t1=time()
    host_data_2x=host_data * np.float32(2)
    t2=time()
    print("total time to compute on CPU: %f" %(t2-t1))

    device_data = gpuarray.to_gpu(host_data)
    # 内核函数涉及c语言中的浮点数类型指针，所以此处需提前分配内存
    # allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data)
    t1=time()
    gpu_2x_ker(device_data, device_data_2x)
    t2=time()
    from_device=device_data_2x.get()
    print ("total time to compute on GPU: %f" % (t2 - t1))
    print ("Is the host computation the same as the GPU computation? : {}".format(np.allclose(from_device, host_data_2x)))


if __name__ == "__main__":
    speedcomparison()
