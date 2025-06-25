import pycuda.autoinit
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as dvr
from time import time

ker = SourceModule(
    """
    __global__ void mult_ker(float *array, int arrayr_len)
    {
        int thd = blockIdx.x * blockDim.x + threadIdx.x;
        int num_iters = arrayr_len / blockDim.x;

        for (int j = 0; j < num_iters; j++)
        {
            int i = j * blockDim.x + thd;

            for (int k = 0; k < 50; k++)
            {
                array[i] *= 2.0;
                array[i] /= 2.0;
            }
        }
    }
"""
)
mult_ker = ker.get_function("mult_ker")

array_len=100*1024**2
data=np.random.randn(array_len).astype("float32")
data_gpu=gpuarray.to_gpu(data)

start_event=dvr.Event()
end_event=dvr.Event()

start_event.record()
mult_ker(data_gpu,np.int32(array_len),block=(64,1,1),grid=(1,1,1))
end_event.record()

end_event.synchronize()

print("Has the kernel started yet? {}".format(start_event.query()))
print("Has the kernel ended yet? {}".format(end_event.query()))

print("Kernel execution time in milliseconds: %f"%start_event.time_till(end_event))