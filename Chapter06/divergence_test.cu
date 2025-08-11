/**
 * @file divergence_test.cu
 * @brief CUDA线程分歧测试程序
 * @details 演示GPU中线程分歧现象，展示不同线程执行不同代码路径对性能的影响
 * @note 线程分歧是GPU编程中的重要概念，影响SIMT执行效率
 * @warning 过多的线程分歧会显著降低GPU性能
 * @todo 可以扩展测试不同分歧模式下的性能差异
 * @code 使用示例
 * nvcc divergence_test.cu -o divergence_test
 * ./divergence_test
 * @endcode
 * @result 使用示例输出结果
 * @example 引用外部示例文件
 * @see CUDA编程指南 - 线程分歧优化章节
 * @since CUDA 1.0
 * @author mekeny1
 * @date 2025-08-12
 * @version 0.1
 * @copyright Copyright (c) 2025
 */

// CUDA运行时库头文件，提供CUDA编程的核心功能
#include<cuda_runtime.h>
// 标准输入输出库，用于printf函数
#include<stdio.h>

/**
 * @brief 线程分歧测试内核函数
 * @details 演示GPU中线程分歧现象的核心测试函数
 * 
 * 线程分歧概念：
 * - 在GPU的SIMT（单指令多线程）执行模型中，同一warp内的线程应该执行相同的指令
 * - 当线程根据条件执行不同代码路径时，称为"线程分歧"
 * - 分歧会导致GPU串行执行不同分支，降低并行效率
 * 
 * 硬件执行机制：
 * - GPU将32个线程组织为一个warp（在大多数GPU上）
 * - 当warp内出现分歧时，GPU会先执行一个分支的所有线程，再执行另一个分支
 * - 这导致某些线程在等待其他线程完成时处于空闲状态
 * 
 * 性能影响：
 * - 分歧程度越高，性能损失越大
 * - 理想情况下，同一warp内的线程应该执行相同的代码路径
 */
__global__ void divergence_test_ker()
{
    // 使用线程ID的奇偶性创建分歧条件
    // threadIdx.x % 2 == 0：偶数线程执行一个分支
    // threadIdx.x % 2 == 1：奇数线程执行另一个分支
    
    if(threadIdx.x%2==0)
    {
        // 偶数线程执行路径：输出偶数线程标识
        // 这个分支会被所有偶数线程（0,2,4,6...）执行
        printf("threadIdx.x %d : This is an even thread.\n",threadIdx.x);
    }
    else
    {
        // 奇数线程执行路径：输出奇数线程标识
        // 这个分支会被所有奇数线程（1,3,5,7...）执行
        printf("threadIdx.x %d : This is an odd thread.\n",threadIdx.x);
    }
    
    // 分歧分析：
    // - 在32线程的warp中，16个线程执行if分支，16个线程执行else分支
    // - GPU会先执行if分支的所有线程，再执行else分支的所有线程
    // - 这导致warp执行效率降低，因为部分线程总是处于等待状态
}

/**
 * @brief 主机端主函数
 * @details 设置GPU设备、启动内核并管理CUDA资源
 * 
 * CUDA编程流程：
 * 1. 选择GPU设备
 * 2. 启动内核函数
 * 3. 同步等待内核完成
 * 4. 清理GPU资源
 */
__host__ int main()
{
    // 选择GPU设备0作为计算设备
    // 在多GPU系统中，可以指定不同的设备ID
    cudaSetDevice(0);
    
    // 启动CUDA内核函数
    // 配置参数：<<<grid_size, block_size>>>
    // - grid_size = 1：启动1个线程块
    // - block_size = 32：每个线程块包含32个线程
    // - 总计：1 × 32 = 32个线程
    // 
    // 线程配置分析：
    // - 32个线程正好组成1个warp（在大多数GPU上）
    // - 这确保了所有线程都在同一个warp中，便于观察分歧现象
    divergence_test_ker<<<1,32>>>();
    
    // 同步等待GPU内核执行完成
    // 这是必要的，因为CUDA内核是异步执行的
    // 主机端需要等待GPU计算完成才能继续后续操作
    cudaDeviceSynchronize();
    
    // 重置GPU设备，清理所有分配的内存和上下文
    // 这是良好的编程习惯，确保程序结束时GPU资源被正确释放
    cudaDeviceReset();
}