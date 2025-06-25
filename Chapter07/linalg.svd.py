import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import misc, linalg
misc.init()

row = 1000
col = 5000

a = np.random.rand(row, col).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)

U_d, s_d, V_d = linalg.svd(a_gpu, lib="cusolver")

U = U_d.get()
s = s_d.get()
V = V_d.get()

S = np.zeros((row, col))
S[:min(row, col), :min(row, col)] = np.diag(s)

print("Can We reconstrut a from its SVD decomposition? :%s" %
      np.allclose(a, np.dot(U, np.dot(S, V)), atol=1e-5))
