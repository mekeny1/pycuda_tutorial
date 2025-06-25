import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas

m = 10
n = 100
alpha = 1
beta = 0
A = np.random.rand(m, n).astype("float32")
x = np.random.rand(n).astype("float32")
y = np.zeros(m).astype("float32")

# 将矩阵A的转置结果保存为副本
A_columnwise = A.T.copy()
A_gpu = gpuarray.to_gpu(A_columnwise)
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)

trans = cublas._CUBLAS_OP['N']

lda = m
incx = 1
incy = 1

handle = cublas.cublasCreate()

cublas.cublasSgemv(handle, trans, m, n, alpha, A_gpu.gpudata,
                   lda, x_gpu.gpudata, incx, beta, y_gpu.gpudata, incy)

# 销毁cuBLAS上下文
cublas.cublasDestroy(handle)
print("cuBLAS returned the correct value: %s" %
      np.allclose(np.dot(A, x), y_gpu.get()))
