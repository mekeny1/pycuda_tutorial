/**
 * @file matrix_ker.cu
 * @brief CUDA矩阵乘法内核实现与性能测试程序
 * @details 实现GPU并行矩阵乘法算法，通过二维线程块结构并行计算矩阵乘积，包含完整的CUDA编程流程和结果验证
 * @note 这是CUDA编程的经典示例，展示了从主机端到设备端的完整开发流程
 * @warning 注意内存管理和同步操作，避免数据竞争和内存泄漏
 * @todo 可以扩展支持不同矩阵尺寸和优化内存访问模式
 * @code 使用示例
 * nvcc matrix_ker.cu -o matrix_ker
 * ./matrix_ker
 * @endcode
 * @result 使用示例输出结果
 * @example 引用外部示例文件
 * @see 相关文档链接
 * @since 版本信息
 * @author mekeny1
 * @date 2025-08-12
 * @version 0.1
 * @copyright Copyright (c) 2025
 */

// CUDA运行时库头文件，提供CUDA编程的核心功能
#include <cuda_runtime.h>
// 标准输入输出库，用于printf函数和错误输出
#include <stdio.h>
// 标准库，提供内存分配函数malloc和free
#include <stdlib.h>

// 数值精度容差定义，用于浮点数比较
#define _EPSILON 0.001
// 绝对值宏定义，用于数值比较
#define _ABS(x) (x>0.0f ? x:-x)

/**
 * @brief 主机端函数：比较两个浮点数组是否在容差范围内相等
 * @details 用于验证GPU计算结果与预期结果的一致性
 * 
 * 数值比较策略：
 * - 使用相对容差_EPSILON进行浮点数比较
 * - 避免浮点数精度问题导致的误判
 * - 返回0表示相等，-1表示不相等
 * 
 * @param A 第一个浮点数组
 * @param B 第二个浮点数组
 * @param len 数组长度
 * @return 0表示相等，-1表示不相等
 */
__host__ int allclose(float *A, float *B,int len)
{
    int return_val=0;

    // 逐元素比较两个数组
    for(int i=0;i<len;i++)
    {
        // 使用容差比较，避免浮点数精度问题
        if(_ABS(A[i]-B[i])>_EPSILON)
        {
            return_val=-1;
            break;
        }
    }

    return return_val;
}

/**
 * @brief 设备端函数：计算矩阵A的第row行与矩阵B的第col列的点积
 * @details 这是矩阵乘法的核心计算单元，每个线程调用一次
 * 
 * 矩阵乘法算法：
 * - 数学公式：C[i,j] = Σ(A[i,k] × B[k,j])，k从0到N-1
 * - 每个线程负责计算输出矩阵的一个元素
 * - 通过循环累加实现点积计算
 * 
 * 内存访问模式：
 * - matrix_a[row*N + k]：访问矩阵A的第row行第k列元素
 * - matrix_b[col + k*N]：访问矩阵B的第k行第col列元素
 * - 行主序存储优化了GPU内存带宽利用率
 * 
 * @param matrix_a 输入矩阵A的GPU内存指针
 * @param matrix_b 输入矩阵B的GPU内存指针
 * @param row 当前线程计算的行索引
 * @param col 当前线程计算的列索引
 * @param N 矩阵维度
 * @return 点积计算结果
 */
__device__ float rowcol_dot(float *matrix_a,float *matrix_b,int row,int col,int N)
{
    float val=0;

    // 遍历矩阵维度N，执行点积计算
    for(int k=0;k<N;k++)
    {
        // 关键内存访问模式：
        // matrix_a[row*N + k]：行主序访问矩阵A
        // matrix_b[col + k*N]：行主序访问矩阵B
        // 这种访问模式优化了GPU内存带宽利用率
        val+=matrix_a[row*N+k]*matrix_b[col+k*N];
    }

    return val;
}

/**
 * @brief 全局内核函数：矩阵乘法的主入口点
 * @details 每个线程负责计算输出矩阵的一个元素
 * 
 * CUDA线程组织：
 * - 使用二维线程块结构(blockDim.x, blockDim.y)
 * - 使用二维网格结构(gridDim.x, gridDim.y)
 * - 每个线程通过blockIdx和threadIdx确定在矩阵中的位置
 * 
 * 线程映射策略：
 * - row = blockIdx.x * blockDim.x + threadIdx.x
 * - col = blockIdx.y * blockDim.y + threadIdx.y
 * - 确保每个矩阵元素都有对应的线程负责计算
 * 
 * @param matrix_a 输入矩阵A的GPU内存指针
 * @param matrix_b 输入矩阵B的GPU内存指针
 * @param output_matirx 输出矩阵C的GPU内存指针
 * @param N 矩阵维度
 */
__global__ void matrix_mult_ker(float *matrix_a,float *matrix_b,float *output_matirx,int N)
{
    // 计算当前线程在矩阵中的全局位置
    // 使用CUDA线程索引系统：全局位置 = 块索引 × 块维度 + 线程索引
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;

    // 将计算结果存储到输出矩阵的对应位置
    // 注意索引计算：col + row*N 确保正确的行主序存储
    // 调用设备端函数计算点积，实现真正的矩阵乘法运算
    output_matirx[col+row*N]=rowcol_dot(matrix_a,matrix_b,row,col,N);
}

/**
 * @brief 主机端主函数：完整的CUDA矩阵乘法程序流程
 * @details 演示从主机端到设备端的完整CUDA编程流程
 * 
 * CUDA编程标准流程：
 * 1. 选择GPU设备
 * 2. 分配主机和设备内存
 * 3. 将数据从主机复制到设备
 * 4. 配置并启动内核
 * 5. 同步等待内核完成
 * 6. 将结果从设备复制回主机
 * 7. 验证结果正确性
 * 8. 释放所有分配的内存
 */
__host__ int main()
{
    // 选择GPU设备0作为计算设备
    // 在多GPU系统中，可以指定不同的设备ID
    cudaSetDevice(0);
    
    // 定义矩阵维度和内存大小
    int N=4;  // 4×4矩阵
    int num_bytes=sizeof(float)*N*N;  // 计算所需字节数

    // 定义主机端输入矩阵A：每行都是[14,13,12,11]
    // 这种测试数据便于调试和验证计算结果
    float h_A[]={ \
        14.0,  13.0,  12.0,  11.0, \
        14.0,  13.0,  12.0,  11.0, \
        14.0,  13.0,  12.0,  11.0, \
        14.0,  13.0,  12.0,  11.0 \
    };

    // 定义主机端输入矩阵B：每行都是[14,13,12,11]
    // 与矩阵A相同，便于预测计算结果
    float h_B[]={ \
        14.0,  13.0,  12.0,  11.0, \
        14.0,  13.0,  12.0,  11.0, \
        14.0,  13.0,  12.0,  11.0, \
        14.0,  13.0,  12.0,  11.0 \
    };

    // 预计算的期望结果矩阵：A × B的预期输出
    // 用于验证GPU计算结果的正确性
    float h_AxB[]={ \
        140.0,  130.0,  120.0,  110.0, \
        140.0,  130.0,  120.0,  110.0, \
        140.0,  130.0,  120.0,  110.0, \
        140.0,  130.0,  120.0,  110.0 \
    };

    // 声明GPU设备内存指针
    float *d_A;      // 矩阵A的GPU内存指针
    float *d_B;      // 矩阵B的GPU内存指针
    float *d_output; // 输出矩阵C的GPU内存指针

    // 在GPU上分配内存空间
    // cudaMalloc：在GPU设备内存中分配指定大小的内存块
    cudaMalloc((float **) &d_A, num_bytes);
    cudaMalloc((float **) &d_B,num_bytes);

    // 将数据从主机内存复制到GPU设备内存
    // cudaMemcpyHostToDevice：主机到设备的数据传输
    // 这是CUDA编程中的关键步骤，确保GPU能够访问输入数据
    cudaMemcpy(d_A,h_A,num_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,num_bytes,cudaMemcpyHostToDevice);

    // 为输出矩阵分配GPU内存空间
    cudaMalloc((float **) &d_output, num_bytes);

    // 在主机端分配内存，用于存储GPU计算结果
    float *h_output;
    h_output=(float *) malloc(num_bytes);
    
    // 配置CUDA线程层次结构
    // dim3：CUDA的三维向量类型，用于定义线程块和网格维度
    // block(2,2,1)：每个线程块包含2×2×1=4个线程
    // grid(2,2,1)：网格包含2×2×1=4个线程块
    // 总计：4×4=16个线程，正好对应4×4矩阵的16个元素
    dim3 block(2,2,1);
    dim3 grid(2,2,1);

    // 启动GPU内核执行矩阵乘法计算
    // 使用三重尖括号语法：<<<grid, block>>>
    // 传递GPU内存指针和矩阵维度参数
    matrix_mult_ker<<<grid,block>>> (d_A,d_B,d_output,N);

    // 同步等待GPU内核执行完成
    // cudaDeviceSynchronize：确保主机端等待GPU计算完成
    // 这是必要的，因为CUDA内核是异步执行的
    cudaDeviceSynchronize();

    // 将计算结果从GPU设备内存复制回主机内存
    // cudaMemcpyDeviceToHost：设备到主机的数据传输
    cudaMemcpy(h_output,d_output,num_bytes,cudaMemcpyDeviceToHost);
    
    // 再次同步，确保数据传输完成
    cudaDeviceSynchronize();

    // 释放GPU设备内存
    // cudaFree：释放通过cudaMalloc分配的设备内存
    // 良好的编程习惯，避免GPU内存泄漏
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);

    // 重置GPU设备，清理所有分配的内存和上下文
    cudaDeviceReset();

    // 验证GPU计算结果与预期结果的一致性
    // allclose函数比较两个数组是否在容差范围内相等
    if(allclose(h_AxB,h_output,N*N)<0)
    {
        // 计算结果不匹配，输出错误信息
        printf("Error!  Output of kernel does not match expected output.\n");
        free(h_output);  // 释放主机内存
        return -1;       // 返回错误码
    }
    else
    {
        // 计算结果匹配，输出成功信息
        printf("Success!  Output of kernel matches expected output.\n");
        free(h_output);  // 释放主机内存
        return 0;        // 返回成功码
    }
}