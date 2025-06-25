import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas

a = np.float32(10)
v = np.float32([1, 2, 3])
w = np.float32([4, 5, 6])

v_gpu = gpuarray.to_gpu(v)
w_gpu = gpuarray.to_gpu(w)

cublas_context_h = cublas.cublasCreate()

dot_output = cublas.cublasSdot(cublas_context_h, v_gpu.size,
                        v_gpu.gpudata, 1, w_gpu.gpudata, 1)

l2_output = cublas.cublasSnrm2(cublas_context_h, v_gpu.size, v_gpu.gpudata, 1)

cublas.cublasDestroy(cublas_context_h)

numpy_dot = np.dot(v, w)
numpy_l2 = np.linalg.norm(v)

print("点积结果是否接近 NumPy 近似值: %s" % np.allclose(dot_output, numpy_dot))
print("L2 范数结果是否接近 NumPy 近似值: %s" % np.allclose(l2_output, numpy_l2))
