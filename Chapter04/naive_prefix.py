import pycuda.autoinit
import pycuda.driver as dvr
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time


naive_ker = SourceModule(
    """
    __global__ void naive_prefix(double *vec, double *out)
    {
        __shared__ double sum_buf[1024];
        int tid = threadIdx.x;
        sum_buf[tid] = vec[tid];

        // begin parallel prefix sum algorithm
        // 以tid=1023为例，iter=1时，值为原始数组中tid=1023与1023-1值的和；iter=2时，再加上tid=1023-2的值，即原始数组tid=1021与1020的和；iter=3时，再加上tid=1023-4的值，即原始数组tid=1019、1018、1017与1016的和；以此类推，每一次循环加上tid=1023-2^i的值，并且这个1023-2^i的值是已经求和原始数组tid=(1023-2^i,1023-2^(i-1)+1)，可以遍历加上tid=1023及之前数组的所有值
        int iter = 1;
        for (int i = 0; i < 10; i++)
        {
            __syncthreads();
            if (tid >= iter)
            {
                sum_buf[tid] = sum_buf[tid] + sum_buf[tid - iter];
            }
            iter *= 2;
        }
        __syncthreads();
        out[tid]=sum_buf[tid];
        __syncthreads();
    }
    """
)


naive_gpu = naive_ker.get_function("naive_prefix")


if __name__ == "__main__":
    testvec = np.random.randn(1024).astype(np.float64)
    testvec_gpu = gpuarray.to_gpu(testvec)
    outvec_gpu = gpuarray.empty_like(testvec_gpu)

    # cuda格网为一个线程块，这个线程块长为1024的线程
    naive_gpu(testvec_gpu, outvec_gpu, block=(1024, 1, 1), grid=(1, 1, 1))

    total_sum = sum(testvec)
    total_sum_gpu = outvec_gpu[-1].get()

    print("Does our kernel work correctly? : {}".format(
        np.allclose(total_sum_gpu, total_sum)))
