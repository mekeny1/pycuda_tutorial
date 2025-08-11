#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel函数：计算Mandelbrot集合
// 每个thread处理一个复数点，判断该点是否属于Mandelbrot集合
extern "C" __global__ void mandelbrot_ker(float *lattice, float *mandelbrot_graph, int max_iters, float upper_bound_squared, int lattice_size)
{
    // 计算当前thread的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保thread索引在有效范围内
    if (tid < lattice_size * lattice_size)
    {
        // 将一维索引转换为二维网格坐标
        int i = tid % lattice_size;                    // x坐标
        int j = lattice_size - 1 - (tid / lattice_size); // y坐标（翻转，使图像正确显示）

        // 获取复平面上的实部和虚部坐标
        float c_re = lattice[i];  // 实部
        float c_im = lattice[j];  // 虚部

        // 初始化迭代变量z = 0 + 0i
        float z_re = 0.0f;
        float z_im = 0.0f;

        // 默认假设点属于Mandelbrot集合（值为1）
        mandelbrot_graph[tid] = 1;

        // 执行Mandelbrot迭代：z = z^2 + c
        for (int k = 0; k < max_iters; k++)
        {
            float temp;

            // 计算z^2的实部：(a+bi)^2 = (a^2-b^2) + 2abi
            temp = z_re * z_re - z_im * z_im + c_re;
            // 计算z^2的虚部
            z_im = 2 * z_re * z_im + c_im;
            z_re = temp;

            // 检查是否发散：|z|^2 > upper_bound_squared
            if ((z_re * z_re + z_im * z_im) > upper_bound_squared)
            {
                // 如果发散，标记为不属于Mandelbrot集合（值为0）
                mandelbrot_graph[tid] = 0;
                break;
            }
        }
    }
    return;
}

// 主机端函数：启动Mandelbrot计算kernel
// 使用__declspec(dllexport)使函数可以从DLL中导出
extern "C" __declspec(dllexport) void launch_mandelbrot(float *lattice, float *mandelbrot_graph, int max_iters, float upper_bound, int lattice_size)
{
    // 计算需要分配的内存大小
    int num_bytes_lattice = sizeof(float) * lattice_size;
    int num_bytes_graph = sizeof(float) * lattice_size * lattice_size;

    // GPU内存指针
    float *d_lattice;
    float *d_mandelbrot_graph;
    
    // 在GPU上分配内存
    cudaMalloc((float **)&d_lattice, num_bytes_lattice);
    cudaMalloc((float **)&d_mandelbrot_graph, num_bytes_graph);

    // 将lattice数据从CPU复制到GPU
    cudaMemcpy(d_lattice, lattice, num_bytes_lattice, cudaMemcpyHostToDevice);

    int grid_size = (int)ceil(((double)lattice_size * lattice_size) / ((double)32));

    // 启动CUDA kernel
    // 使用32个线程每block，grid_size个block
    mandelbrot_ker<<<grid_size, 32>>>(d_lattice, d_mandelbrot_graph, max_iters, upper_bound * upper_bound, lattice_size);

    // 将计算结果从GPU复制回CPU
    cudaMemcpy(mandelbrot_graph, d_mandelbrot_graph, num_bytes_graph, cudaMemcpyDeviceToHost);

    // 释放GPU内存
    cudaFree(d_lattice);
    cudaFree(d_mandelbrot_graph);
}
