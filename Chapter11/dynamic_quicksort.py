import numpy as np
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit
from pycuda import gpuarray
from random import shuffle

# CUDA C 代码，演示动态并行实现的快速排序
DynamicQuicksortCode = """
// 分区函数，返回枢轴位置
__device__ int partition(int *a, int lo, int hi)
{
    int i = lo;
    int pivot = a[hi];
    int temp;

    for (int k = lo; k < hi; k++)
    {
        if (a[k] < pivot)
        {
            temp = a[k];
            a[k] = a[i];
            a[i] = temp;
            i++;
        }
    }

    a[hi] = a[i];
    a[i] = pivot;

    return i;
}

// 动态并行快速排序核函数
__global__ void quicksort_ker(int *a, int lo, int hi)
{
    // 创建两个非阻塞 CUDA 流，分别递归左右子区间
    cudaStream_t s_left, s_right;
    cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_right, cudaStreamNonBlocking);

    int mid = partition(a, lo, hi); // 分区
    
    // 递归排序左半部分
    if (mid - 1 - lo > 0)
        quicksort_ker<<<1, 1, 0, s_left>>>(a, lo, mid - 1);
    // 递归排序右半部分
    if (hi - (mid + 1) > 0)
        quicksort_ker<<<1, 1, 0, s_right>>>(a, mid + 1, hi);
    // 销毁流
    cudaStreamDestroy(s_left);
    cudaStreamDestroy(s_right);
}
"""

# 编译动态并行 CUDA 代码，获取核函数
qsort_mod = DynamicSourceModule(DynamicQuicksortCode)
qsort_ker = qsort_mod.get_function("quicksort_ker")

if __name__ == "__main__":
    # 生成乱序数组
    a = list(range(1000))
    shuffle(a)
    a = np.int32(a)
    d_a = gpuarray.to_gpu(a)  # 拷贝到 GPU
    print("Unsorted array: %s" % a)
    # 启动动态并行快速排序核函数
    qsort_ker(d_a, np.int32(0), np.int32(a.size-1),
              grid=(1, 1, 1), block=(1, 1, 1))
    a_sorted = list(d_a.get())  # 拷贝回主机
    print("Sorted array: %s" % a_sorted)
