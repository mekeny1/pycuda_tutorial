import pycuda.autoinit
import pycuda.driver as dvr
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule


ker=SourceModule(
    """
    __global__ void scalar_multiply_kernel(float *outVec, float scalar, float *vec)
    {
        int i=threadIdx.x;
        outVec[i]=scalar*vec[i];
    }
    """
)


scalar_multiply_gpu=ker.get_function("scalar_multiply_kernel")
testVec=np.random.randn(512).astype(np.float32)
testVec_gpu=gpuarray.to_gpu(testVec)
outVec_gpu=gpuarray.empty_like(testVec_gpu)

scalar_multiply_gpu(outVec_gpu,np.float32(2),testVec_gpu,block=(512,1,1),grid=(1,1,1,))

print("Does our kernel work correctly? : {}".format(np.allclose(outVec_gpu.get(),2*testVec)))
