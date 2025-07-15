import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# CUDA C 代码，演示向量化内存访问（int4, double2）
VecCode = """
__global__ void vec_ker(int *ints, double *doubles)
{
    int4 f1, f2;

    // 通过 int4 类型一次性读取 4 个 int
    f1 = *reinterpret_cast<int4 *>(ints);
    f2 = *reinterpret_cast<int4 *>(&ints[4]);

    printf("First int4: %d, %d, %d, %d\\n", f1.x, f1.y, f1.z, f1.w);
    printf("Second int4: %d, %d, %d, %d\\n", f2.x, f2.y, f2.z, f2.w);

    double2 d1, d2;

    // 通过 double2 类型一次性读取 2 个 double
    d1 = *reinterpret_cast<double2 *>(doubles);
    d2 = *reinterpret_cast<double2 *>(&doubles[2]);

    printf("First double2: %f, %f\\n", d1.x, d1.y);
    printf("Second double2: %f, %f\\n", d2.x, d2.y);
}
"""

# 编译 CUDA 代码，获取核函数
vec_mod = SourceModule(VecCode)
vec_ker = vec_mod.get_function("vec_ker")

# 构造输入数据，8 个 int 和 4 个 double，并拷贝到 GPU
ints = gpuarray.to_gpu(np.int32([1, 2, 3, 4, 5, 6, 7, 8]))
doubles = gpuarray.to_gpu(np.double([1.11, 2.22, 3.33, 4.44]))

print("Vectorized Memory Test:")
# 启动核函数，测试向量化内存访问
vec_ker(ints, doubles, grid=(1, 1, 1), block=(1, 1, 1))
