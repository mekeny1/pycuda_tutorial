'''
Author: mekeny1
Date: 2025-07-12 17:01:44
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:29:20
FilePath: \pycuda_tutorial_hapril\Chapter11\dynamic_quicksort.py
Description: CUDA动态并行快速排序实现，展示GPU上的递归排序算法和CUDA流并行处理
Tags: cuda, dynamic-parallelism, quicksort, gpu-sorting, cuda-streams, recursive-algorithm, parallel-computing
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import numpy as np
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit
from pycuda import gpuarray
from random import shuffle

# CUDA C 代码，演示动态并行实现的快速排序
# 这是GPU上实现经典快速排序算法的动态并行版本
DynamicQuicksortCode = """
// 分区函数，返回枢轴位置
// 这是快速排序的核心操作，将数组分为小于枢轴和大于枢轴的两部分
__device__ int partition(int *a, int lo, int hi)
{
    int i = lo;
    int pivot = a[hi];  // 选择最后一个元素作为枢轴
    int temp;

    // 遍历数组，将小于枢轴的元素移到左边
    for (int k = lo; k < hi; k++)
    {
        if (a[k] < pivot)
        {
            // 交换元素，将小于枢轴的元素移到位置i
            temp = a[k];
            a[k] = a[i];
            a[i] = temp;
            i++;
        }
    }

    // 将枢轴放到正确位置
    a[hi] = a[i];
    a[i] = pivot;

    return i;  // 返回枢轴的最终位置
}

// 动态并行快速排序核函数
// 使用CUDA流实现左右子区间的并行递归排序
__global__ void quicksort_ker(int *a, int lo, int hi)
{
    // 创建两个非阻塞 CUDA 流，分别递归左右子区间
    // cudaStreamNonBlocking标志确保流的创建不会阻塞当前线程
    cudaStream_t s_left, s_right;
    cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_right, cudaStreamNonBlocking);

    int mid = partition(a, lo, hi); // 分区，获取枢轴位置
    
    // 递归排序左半部分
    // 只有当左半部分有多个元素时才进行递归
    if (mid - 1 - lo > 0)
        quicksort_ker<<<1, 1, 0, s_left>>>(a, lo, mid - 1);
    // 递归排序右半部分
    // 只有当右半部分有多个元素时才进行递归
    if (hi - (mid + 1) > 0)
        quicksort_ker<<<1, 1, 0, s_right>>>(a, mid + 1, hi);
    // 销毁流，释放GPU资源
    cudaStreamDestroy(s_left);
    cudaStreamDestroy(s_right);
}
"""

# 编译动态并行 CUDA 代码，获取核函数
# DynamicSourceModule支持内核函数间的相互调用，这是动态并行的基础
qsort_mod = DynamicSourceModule(DynamicQuicksortCode)
qsort_ker = qsort_mod.get_function("quicksort_ker")

if __name__ == "__main__":
    # 生成乱序数组
    # 创建1-1000的数组并随机打乱，用于测试排序算法
    a = list(range(1000))
    shuffle(a)
    a = np.int32(a)
    d_a = gpuarray.to_gpu(a)  # 拷贝到 GPU
    print("Unsorted array: %s" % a)
    # 启动动态并行快速排序核函数
    # 传入数组、起始索引和结束索引，使用单个线程启动排序
    qsort_ker(d_a, np.int32(0), np.int32(a.size-1),
              grid=(1, 1, 1), block=(1, 1, 1))
    a_sorted = list(d_a.get())  # 拷贝回主机
    print("Sorted array: %s" % a_sorted)
