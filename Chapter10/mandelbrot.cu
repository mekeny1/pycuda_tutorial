/**
 * @file mandelbrot.cu
 * @brief Mandelbrot集合CUDA内核实现
 * @details 使用CUDA并行计算实现Mandelbrot分形集合的高效生成
 * @note 每个GPU线程处理一个复数点，实现大规模并行计算
 * @warning 需要NVIDIA GPU和CUDA运行时环境
 * @todo 可优化内存访问模式，增加更多分形算法
 * @code 使用示例
 * // 编译为PTX文件：nvcc -ptx mandelbrot.cu -o mandelbrot.ptx
 * // 编译为DLL文件：nvcc --shared -Xcompiler -fPIC mandelbrot.cu -o mandelbrot.dll
 * @endcode
 * @result 生成包含每个点迭代次数的2D数组，用于分形图像可视化
 * @example 参考mandelbrot_driver.py和mandelbrot_ptx.py的使用示例
 * @see CUDA Programming Guide, NVIDIA CUDA Toolkit Documentation
 * @since CUDA 1.0
 * @author mekeny1
 * @date 2025-08-13
 * @version 0.1
 * @copyright Copyright (c) 2025
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==================== CUDA内核函数 ====================

// CUDA kernel函数：计算Mandelbrot集合
// 每个thread处理一个复数点，判断该点是否属于Mandelbrot集合
// @parallel_strategy: 使用GPU的数千个线程同时计算不同复数点
// @memory_access: 每个线程独立访问全局内存，无线程间依赖
extern "C" __global__ void mandelbrot_ker(float *lattice, float *mandelbrot_graph, int max_iters, float upper_bound_squared, int lattice_size)
{
    // 计算当前thread的全局索引
    // @thread_identification: 使用block和thread索引计算全局线程ID
    // @work_distribution: 每个线程负责计算一个复数点的迭代结果
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保thread索引在有效范围内
    // @bounds_checking: 防止线程访问超出数组边界的内存
    if (tid < lattice_size * lattice_size)
    {
        // 将一维索引转换为二维网格坐标
        // @coordinate_mapping: 将线性线程ID映射到复数平面的2D坐标
        // @image_orientation: 翻转Y坐标确保图像正确显示
        int i = tid % lattice_size;                    // x坐标（实部）
        int j = lattice_size - 1 - (tid / lattice_size); // y坐标（虚部，翻转显示）

        // 获取复平面上的实部和虚部坐标
        // @complex_plane: 从预计算的网格中获取复数c的实部和虚部
        float c_re = lattice[i];  // 复数c的实部
        float c_im = lattice[j];  // 复数c的虚部

        // 初始化迭代变量z = 0 + 0i
        // @iteration_start: 从原点开始Mandelbrot迭代过程
        float z_re = 0.0f;  // z的实部初始值
        float z_im = 0.0f;  // z的虚部初始值

        // 默认假设点属于Mandelbrot集合（值为1）
        // @initial_assumption: 假设点属于集合，迭代过程中可能被排除
        mandelbrot_graph[tid] = 1;

        // 执行Mandelbrot迭代：z = z^2 + c
        // @fractal_algorithm: 核心分形算法，判断点是否属于Mandelbrot集合
        // @convergence_test: 通过迭代判断序列是否收敛或发散
        for (int k = 0; k < max_iters; k++)
        {
            float temp;

            // 计算z^2的实部：(a+bi)^2 = (a^2-b^2) + 2abi
            // @complex_multiplication: 复数平方的数学实现
            // @optimization: 使用临时变量避免重复计算
            temp = z_re * z_re - z_im * z_im + c_re;  // 新z的实部 = (z_re^2 - z_im^2) + c_re
            // 计算z^2的虚部
            z_im = 2 * z_re * z_im + c_im;  // 新z的虚部 = 2*z_re*z_im + c_im
            z_re = temp;

            // 检查是否发散：|z|^2 > upper_bound_squared
            // @divergence_test: 使用模的平方避免开方运算，提升性能
            // @early_termination: 一旦发散立即退出循环，节省计算资源
            if ((z_re * z_re + z_im * z_im) > upper_bound_squared)
            {
                // 如果发散，标记为不属于Mandelbrot集合（值为0）
                // @set_membership: 发散的点不属于Mandelbrot集合
                mandelbrot_graph[tid] = 0;
                break;  // 提前退出迭代循环
            }
        }
    }
    return;
}

// ==================== 主机端函数 ====================

// 主机端函数：启动Mandelbrot计算kernel
// 使用__declspec(dllexport)使函数可以从DLL中导出
// @host_interface: 提供Python等高级语言调用的C接口
// @memory_management: 管理GPU内存的分配、传输和释放
extern "C" __declspec(dllexport) void launch_mandelbrot(float *lattice, float *mandelbrot_graph, int max_iters, float upper_bound, int lattice_size)
{
    // 计算需要分配的内存大小
    // @memory_calculation: 根据数据大小计算GPU内存需求
    int num_bytes_lattice = sizeof(float) * lattice_size;  // 输入网格内存大小
    int num_bytes_graph = sizeof(float) * lattice_size * lattice_size;  // 输出结果内存大小

    // GPU内存指针
    // @device_memory: 声明GPU内存指针，用于存储输入和输出数据
    float *d_lattice;  // GPU上的输入网格数据
    float *d_mandelbrot_graph;  // GPU上的输出结果数据
    
    // 在GPU上分配内存
    // @memory_allocation: 为输入和输出数据分配GPU全局内存
    cudaMalloc((float **)&d_lattice, num_bytes_lattice);
    cudaMalloc((float **)&d_mandelbrot_graph, num_bytes_graph);

    // 将lattice数据从CPU复制到GPU
    // @host_to_device: 将输入数据从主机内存传输到GPU内存
    // @data_transfer: 异步内存传输，启动GPU计算前的数据准备
    cudaMemcpy(d_lattice, lattice, num_bytes_lattice, cudaMemcpyHostToDevice);

    // 计算线程网格配置
    // @grid_configuration: 根据数据大小和线程块大小计算最优网格配置
    // @parallel_efficiency: 确保所有GPU核心都有足够的工作负载
    int grid_size = (int)ceil((double)(lattice_size * lattice_size) / 32.0);

    // 启动CUDA kernel
    // 使用32个线程每block，grid_size个block
    // @kernel_launch: 在GPU上启动并行Mandelbrot计算
    // @thread_configuration: 配置线程块大小(32)和网格大小
    // @parameter_passing: 传递所有必要的参数给内核函数
    mandelbrot_ker<<<grid_size, 32>>>(d_lattice, d_mandelbrot_graph, max_iters, upper_bound * upper_bound, lattice_size);

    // 将计算结果从GPU复制回CPU
    // @device_to_host: 将GPU计算结果传输回主机内存
    // @result_retrieval: 获取分形图像数据用于后续处理和可视化
    cudaMemcpy(mandelbrot_graph, d_mandelbrot_graph, num_bytes_graph, cudaMemcpyDeviceToHost);

    // 释放GPU内存
    // @memory_cleanup: 释放之前分配的GPU内存，防止内存泄漏
    // @resource_management: 确保GPU资源正确释放
    cudaFree(d_lattice);
    cudaFree(d_mandelbrot_graph);
}
