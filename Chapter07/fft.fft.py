import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft

N=1000
x = np.asarray(np.random.rand(N), dtype=np.float32)
x_gpu = gpuarray.to_gpu(x)
x_hat = gpuarray.empty_like(x_gpu, dtype=np.complex64)

plan = fft.Plan(x_gpu.shape, np.float32, np.complex64)

inverse_plan = fft.Plan(x.shape, in_dtype=np.complex64, out_dtype=np.float32)

fft.fft(x_gpu, x_hat, plan)
fft.ifft(x_hat, x_gpu, inverse_plan, scale=True)

y = np.fft.fft(x)
# cuFFT在处理实数时，只计算前半部分的输出，其余部分设为0，而NumPy不会
# print("cuFFT matches Numpy FFT: %s" % np.allclose(x_hat.get(), y, atol=1e-6))
print("cuFFT matches Numpy FFT: %s" % np.allclose(
    x_hat.get()[0:N//2], y[0:N//2], atol=1e-6))
print("cuFFT inverse matches original: %s" %
      np.allclose(x_gpu.get(), x, atol=1e-6))
