import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# CUDA C 代码，演示 PTX 内联汇编的各种用法
PtxCode = """
extern "C" {
    // 将 int 变量置零
    __device__ void set_to_zero(int *x)
    {
        asm("mov.s32 %0, 0;" : "=r"(*x));
    }

    // 执行 float 加法，结果写入 out
    __device__ void add_floats(float *out, float in1, float in2)
    {
        asm("add.f32 %0, %1, %2;" : "=f"(*out) : "f"(in1), "f"(in2));
    }

    // int 自增
    __device__ void plusplus(int *x)
    {
        asm("add.s32 %0, %0, 1;" : "+r"(*x));
    }

    // 获取当前线程的 lane id
    __device__ int laneid()
    {
        int id;
        asm("mov.u32 %0, %%laneid;" : "=r"(id));
        return id;
    }

    // 将 double 拆分为两个 int（低32位和高32位）
    __device__ void split64(double val, int *lo, int *hi)
    {
        asm volatile("mov.b64 {%0,%1},%2;" : "=r"(*lo), "=r"(*hi) : "d"(val));
    }

    // 将两个 int 合成为 double
    __device__ void combined64(double *val, int lo, int hi)
    {
        asm volatile("mov.b64 %0,{%1,%2};" : "=d"(*val) : "r"(lo), "r"(hi));
    }

    // 主核函数，演示各种 PTX 内联汇编用法
    __global__ void ptx_test_ker()
    {
        int x = 123;
        printf("x is %d \\n", x);

        set_to_zero(&x); // 置零
        printf("x is now %d \\n", x);

        plusplus(&x); // 自增
        printf("x is now %d \\n", x);

        float f = 0.0f;
        printf("f is %f \\n", f);

        add_floats(&f, 1.11f, 2.22f); // 浮点加法
        printf("f is now %f \\n", f);

        printf("lane ID is %d \\n", laneid()); // 打印 lane id

        double orig = 3.1415;
        int t1 = 0, t2 = 0;
        split64(orig, &t1, &t2); // double 拆分

        double recon = 0.0;
        combined64(&recon, t1, t2); // 合成 double
        printf("Do split64 / combined64 work? : %s \\n", (orig == recon) ? "true" : "false");
    }
}
"""

# 编译 CUDA 代码，获取核函数
ptx_mod = SourceModule(PtxCode)
ptx_test_ker = ptx_mod.get_function("ptx_test_ker")

# 启动核函数，单线程测试 PTX 汇编功能
ptx_test_ker(grid=(1, 1, 1), block=(1, 1, 1))
