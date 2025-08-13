'''
Author: mekeny1
Date: 2025-06-28 17:04:36
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:17:49
FilePath: \pycuda_tutorial_hapril\Chapter08\monte_carlo_integrater.py
Description: Monte Carlo积分器 - 基于CUDA的数值积分实现
    本模块实现了一个基于CUDA的Monte Carlo数值积分器，利用GPU并行计算能力
    进行高精度的数值积分计算。该实现支持自定义数学函数，并能够处理单精度
    和双精度浮点数计算。
Core Features:
    - GPU并行Monte Carlo积分算法
    - 支持自定义数学函数表达式
    - 单精度/双精度浮点数支持
    - CUDA随机数生成器集成
    - 可配置的线程块和采样参数
Algorithm:
    - 将积分区间分割为多个子区间
    - 每个GPU线程负责一个子区间的随机采样
    - 使用curand库生成高质量随机数
    - 并行计算函数值并累加结果
    - 最终通过密度归一化得到积分值
Hardware Requirements:
    - NVIDIA GPU with CUDA support
    - CUDA Toolkit installed
    - PyCUDA library
Software Dependencies:
    - pycuda.autoinit: CUDA上下文初始化
    - pycuda.driver: CUDA驱动接口
    - pycuda.gpuarray: GPU数组操作
    - pycuda.compiler: CUDA内核编译
    - numpy: 数值计算支持
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np

# CUDA内核模板 - 定义Monte Carlo积分的核心计算逻辑
# 使用模板字符串支持不同精度和数学函数
MonteCarloKernelTemplate = """
    #include <curand_kernel.h>

    #define ULL unsigned long long
    #define _R(z) (1.0f / (z))  // 倒数宏定义
    #define _P2(z) ((z) * (z))  // 平方宏定义

    // 设备端数学函数 - 由用户定义的数学表达式
    __device__ inline %(p)s f(%(p)s x)
    {
        %(p)s y;
        %(math_function)s;  // 用户定义的数学函数表达式

        return y;
    }

    extern "C"
    {
        // Monte Carlo积分内核函数
        // 每个线程负责一个子区间的随机采样计算
        __global__ void monte_carlo(int iters, %(p)s lo, %(p)s hi, %(p)s * ys_out)
        {
            curandState cr_state;  // CUDA随机数生成器状态
            int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程ID
            int num_threads = blockDim.x * gridDim.x;  // 总线程数

            // 计算每个线程负责的子区间宽度
            %(p)s t_width = (hi - lo) / (%(p)s) num_threads;

            // 计算采样密度 - 用于最终结果归一化
            %(p)s density = ((%(p)s) iters) / t_width;

            // 计算当前线程负责的积分区间 [t_lo, t_hi]
            %(p)s t_lo = t_width * tid + lo;
            %(p)s t_hi = t_lo + t_width;

            // 初始化随机数生成器 - 使用时钟和线程ID确保随机性
            curand_init((ULL)clock() + (ULL)tid, (ULL)0, (ULL)0, &cr_state);

            %(p)s y, y_sum = 0.0f;  // 累加变量
            %(p)s rand_val, x;      // 随机数和采样点

            // Monte Carlo采样循环 - 每个线程进行iters次随机采样
            for (int i = 0; i < iters; i++)
            {
                // 生成[0,1)区间内的均匀随机数
                rand_val = curand_uniform%(p_curand)s(&cr_state);
                // 将随机数映射到当前线程的积分区间
                x = t_lo + t_width * rand_val;
                // 计算函数值并累加
                y_sum += f(x);
            }

            // 计算当前线程的积分贡献 - 通过密度归一化
            y = y_sum / density;

            // 将结果存储到全局内存
            ys_out[tid]=y;
        }
    }
"""


class MonteCarloIntegrator:
    """
    Monte Carlo积分器类

    实现基于CUDA的并行Monte Carlo数值积分算法，支持自定义数学函数
    和高精度浮点数计算。

    Attributes:
        math_function (str): 用户定义的数学函数表达式
        precision (str): 计算精度 ('float' 或 'double')
        numpy_precision: 对应的numpy数据类型
        p_curand (str): curand函数精度后缀
        lo (float): 积分下限
        hi (float): 积分上限
        ker: 编译后的CUDA内核对象
        f: CUDA内核函数引用
        num_blocks (int): GPU线程块数量
        samples_per_thread (int): 每个线程的采样次数
    """

    def __init__(self, math_function="y=sin(x)", precision='d', lo=0, hi=np.pi, samples_per_thread=10**5, num_blocks=100):
        """
        初始化Monte Carlo积分器

        Args:
            math_function (str): 数学函数表达式，使用C语法
            precision (str): 计算精度 ('s'/'S'/'single' 为单精度, 'd'/'D'/'double' 为双精度)
            lo (float): 积分下限
            hi (float): 积分上限
            samples_per_thread (int): 每个GPU线程的采样次数
            num_blocks (int): GPU线程块数量
        """
        self.math_function = math_function

        # 精度设置 - 支持单精度和双精度浮点数
        if precision in [None, 's', 'S', "single", np.float32]:
            self.precision = "float"
            self.numpy_precision = np.float32
            self.p_curand = ""  # 单精度curand函数无后缀
        elif precision in ['d', 'D', "double", np.float64]:
            self.precision = "double"
            self.numpy_precision = np.float64
            self.p_curand = "_double"  # 双精度curand函数后缀
        else:
            raise Exception("precision is invalid datatype!")

        # 验证积分区间
        if hi-lo <= 0:
            raise Exception("hi-lo<=0!")
        else:
            self.hi = hi
            self.lo = lo

        # 构建CUDA内核代码 - 替换模板中的占位符
        MonteCarloDict = {"p": self.precision,
                          "p_curand": self.p_curand, "math_function": self.math_function}

        self.MonteCarloCode = MonteCarloKernelTemplate % MonteCarloDict

        # 编译CUDA内核代码
        # no_extern_c=True: 允许使用C++语法
        # options=["-w"]: 抑制警告信息
        self.ker = SourceModule(no_extern_c=True, options=[
                                "-w"], source=self.MonteCarloCode)

        # 获取编译后的内核函数引用
        self.f = self.ker.get_function("monte_carlo")
        self.num_blocks = num_blocks
        self.samples_per_thread = samples_per_thread

    def definite_integral(self, lo=None, hi=None, samples_per_thread=None, num_blocks=None):
        """
        计算定积分值

        执行GPU并行的Monte Carlo积分计算，将积分区间分割给多个GPU线程
        并行处理，最后汇总所有线程的计算结果。

        Args:
            lo (float, optional): 积分下限，默认使用初始化时的值
            hi (float, optional): 积分上限，默认使用初始化时的值
            samples_per_thread (int, optional): 每个线程采样次数
            num_blocks (int, optional): GPU线程块数量

        Returns:
            float: 计算得到的积分值
        """
        # 参数默认值处理
        if lo is None or hi is None:
            lo = self.lo
            hi = self.hi

        if samples_per_thread is None:
            samples_per_thread = self.samples_per_thread

        if num_blocks is None:
            num_blocks = self.num_blocks
            grid = (num_blocks, 1, 1)  # 一维网格配置
        else:
            grid = (num_blocks, 1, 1)

        # GPU线程配置 - 每个块32个线程
        block = (32, 1, 1)
        num_threads = 32*num_blocks

        # 在GPU上分配结果存储数组
        self.ys = gpuarray.empty((num_threads,), dtype=self.numpy_precision)

        # 启动CUDA内核 - 执行并行Monte Carlo计算
        # 参数顺序: 采样次数, 积分下限, 积分上限, 结果数组
        self.f(np.int32(samples_per_thread), self.numpy_precision(lo),
               self.numpy_precision(hi), self.ys, block=block, grid=grid)

        # 将GPU结果传输到CPU并计算总和
        self.nintegral = np.sum(self.ys.get())

        return np.sum(self.nintegral)


if __name__ == "__main__":
    # 测试用例 - 包含复杂数学函数的积分测试
    integral_tests = [
        ("y=log(x)*_P2(sin(x))", 11.733, 18.472, 8.9999),  # 对数与正弦函数组合
        ("y=_R(1+sinh(2*x)*_P2(log(x)))", .9, 4, .584977),  # 双曲正弦与对数函数组合
        ("y=(cosh(x)*sin(x))/sqrt(pow(x,3)+_P2(sin(x)))", 1.85, 4.81, -3.34553)]  # 双曲余弦与正弦函数组合

    # 执行所有测试用例
    for f, lo, hi, expected in integral_tests:
        # 创建积分器实例 - 使用双精度计算
        mci = MonteCarloIntegrator(
            math_function=f, precision="d", lo=lo, hi=hi)

        # 计算积分值并输出结果
        print("The Monte Carlo numercal integration of the function\n \t f: x -> %s \n \t from x = %s to x= %s is: %s" %
              (f, lo, hi, mci.definite_integral()))
        print("where the expected value is: %s\n" % expected)
