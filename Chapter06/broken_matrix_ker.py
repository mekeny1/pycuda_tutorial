'''
Author: mekeny1
Date: 2025-06-17 13:45:30
LastEditors: mekeny1
LastEditTime: 2025-08-11 16:02:33
FilePath: \pycuda_tutorial_hapril\Chapter06\02.broken_matrix_ker.py
Description: 使用 PyCUDA 实现 GPU 矩阵乘法内核，通过二维线程块并行计算矩阵乘积，并验证结果正确性
#algorithm: GPU并行矩阵乘法算法实现
#cuda: CUDA内核编程与线程块管理
#hardware: GPU内存访问优化与并行计算
#performance: 矩阵乘法性能优化策略
#verification: GPU与CPU结果一致性验证
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
# PyCUDA自动初始化，设置CUDA上下文和内存管理
import pycuda.autoinit
# CUDA驱动程序接口，用于底层GPU操作控制
import pycuda.driver as drv
# GPU数组类，提供主机与GPU内存间的数据传输接口
from pycuda import gpuarray
# CUDA源码编译器，将CUDA C代码编译为可执行的GPU内核
from pycuda.compiler import SourceModule
# NumPy数值计算库，用于矩阵运算和结果验证
import numpy as np

"""
- 目标：实现 GPU 并行矩阵乘法，通过二维线程块结构并行计算矩阵乘积，并验证 GPU 计算结果与 CPU 参考结果的一致性。
- 核心算法：
  1) 矩阵乘法 C = A × B：每个输出元素 C[i,j] = Σ(A[i,k] × B[k,j])，k 从 0 到 N-1。
  2) 并行策略：每个线程负责计算一个输出矩阵元素，通过二维线程块索引 (threadIdx.x, threadIdx.y) 和块索引 (blockIdx.x, blockIdx.y) 确定线程在矩阵中的位置。
  3) 内存访问模式：行主序存储，A 矩阵按行访问，B 矩阵按列访问，适合 GPU 内存带宽优化。
- CUDA/硬件要点：
  - 使用二维线程块结构 (blockDim.x, blockDim.y) 和二维网格 (gridDim.x, gridDim.y)，每个线程计算矩阵的一个元素。
  - 通过 blockIdx*blockDim + threadIdx 计算全局线程坐标，映射到矩阵的行列索引。
  - 设备端函数 rowcol_dot 执行点积计算，全局内核 matrix_mult_ker 负责线程分配和结果存储。
  - 使用 gpuarray 进行主机-设备数据传输，支持自动内存管理和类型转换。
- 测试配置：4×4 矩阵，线程块 (2,2,1)，网格 (2,2,1)，共 16 个线程对应 16 个矩阵元素。
"""

# 编译CUDA源码为可执行的GPU模块
# 包含两个关键函数：设备端点积计算函数和全局矩阵乘法内核
ker = SourceModule(
    """
    // 设备端函数：计算矩阵A的第row行与矩阵B的第col列的点积
    // 这是矩阵乘法的核心计算单元，每个线程调用一次
    __device__ float rowcol_dot(float *matrix_a, float *matrix_b, int row, int col, int N)
    {
        // printf("threadIdx.x,y: %d,%d blockIdx.x,y: %d,%d -- row is %d, col is %d, N is %d.\\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col, N);

        // 初始化点积结果
        float val = 0;

        // 遍历矩阵维度N，执行点积计算
        // 矩阵乘法公式：C[i,j] = Σ(A[i,k] × B[k,j])，k从0到N-1
        for (int k = 0; k < N; k++)
        {
            // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                // printf("Dot-product loop: k value is %d, matrix_a value is %f, matrix_b is %f.\\n", k, matrix_a[ row + k*N ], matrix_b[ col*N + k]); 

            // 关键内存访问模式：
            // matrix_a[row*N + k]: 访问矩阵A的第row行第k列元素（行主序存储）
            // matrix_b[col + k*N]: 访问矩阵B的第k行第col列元素（行主序存储）
            // 这种访问模式优化了GPU内存带宽利用率
            val += matrix_a[row*N + k] * matrix_b[col + k*N];
        }

        return val;
    }

    // 全局内核函数：矩阵乘法的主入口点
    // 每个线程负责计算输出矩阵的一个元素
    __global__ void matrix_mult_ker(float *matrix_a, float *matrix_b,float *output_matrix, int N)
    {
        // 计算当前线程在矩阵中的全局位置
        // 使用CUDA线程索引系统：全局位置 = 块索引 × 块维度 + 线程索引
        // 这确保了每个矩阵元素都有对应的线程负责计算
        int row =blockIdx.x*blockDim.x+threadIdx.x;
        int col =blockIdx.y*blockDim.y+threadIdx.y;

        // printf("threadIdx.x,y: %d, %d; blockIdx.x,y: %d, %d; -- row is %d, col is %d.\\n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,row,col);

        // 将计算结果存储到输出矩阵的对应位置
        // 注意索引计算：col + row*N 确保正确的行主序存储
        // 调用设备端函数计算点积，实现真正的矩阵乘法运算
        output_matrix[col+row*N]=rowcol_dot(matrix_a,matrix_b,row,col,N);
    }
"""
)

# 从编译好的CUDA模块中获取矩阵乘法内核函数的Python调用句柄
# 这个句柄用于在Python中调用GPU内核函数
matrix_ker = ker.get_function("matrix_mult_ker")

# 创建测试数据：4×4矩阵用于验证算法正确性
# test_a: 每行都是[1,2,3,4]的递增序列，便于调试和验证
# test_b: 每行都是[14,13,12,11]的递减序列，便于调试和验证
test_a = np.float32(list([range(1, 5)])*4)
test_b = np.float32(list([range(14, 10, -1)])*4)

# 使用NumPy的matmul函数计算CPU参考结果
# 这是标准实现，用于验证GPU计算结果的正确性
output_mat = np.matmul(test_a, test_b)

# GPU内存管理：将输入矩阵传输到GPU，为输出矩阵分配GPU内存空间
# gpuarray.to_gpu(): 将主机内存数据复制到GPU设备内存
# gpuarray.empty_like(): 在GPU上分配与输入矩阵相同形状的空矩阵
test_a_gpu = gpuarray.to_gpu(test_a)
test_b_gpu = gpuarray.to_gpu(test_b)
output_mat_gpu = gpuarray.empty_like(test_a_gpu)

# 发射GPU内核执行矩阵乘法计算
# 参数说明：
# - 前三个参数：GPU上的输入矩阵A、B和输出矩阵C
# - np.int32(4): 矩阵维度N=4
# - block=(2,2,1): 每个线程块包含2×2×1=4个线程
# - grid=(2,2,1): 网格包含2×2×1=4个线程块
# - 总计：4×4=16个线程，正好对应4×4矩阵的16个元素
matrix_ker(test_a_gpu, test_b_gpu, output_mat_gpu,
           np.int32(4), block=(2, 2, 1), grid=(2, 2, 1))

# 结果验证：使用assert确保GPU计算结果与CPU参考结果在数值精度范围内一致
# np.allclose(): 检查两个数组是否在相对和绝对容差范围内相等
# 这是GPU计算正确性的关键验证步骤
assert (np.allclose(output_mat_gpu.get(), output_mat))

# 输出验证结果：True表示GPU计算正确，False表示存在计算错误
# 这个输出帮助开发者快速判断算法实现是否成功
print(np.allclose(output_mat_gpu.get(), output_mat))
