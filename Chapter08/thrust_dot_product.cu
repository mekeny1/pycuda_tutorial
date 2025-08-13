/**
 * @file thrust_dot_product.cu
 * @brief Thrust点积计算演示 - 使用Thrust库实现GPU并行向量点积运算
 * @details 本文件演示了如何使用Thrust库实现向量点积运算。点积是向量运算中的
 *          基本操作，定义为两个向量对应元素相乘后求和。本实现利用Thrust的
 *          transform和reduce算法，在GPU上高效并行计算点积。
 * @note 点积运算在机器学习、信号处理、图形学等领域有广泛应用。Thrust库的
 *       transform-reduce模式是并行计算中的经典模式，适用于多种向量运算。
 * @warning 需要确保CUDA Toolkit和Thrust库已正确安装，输入向量长度必须相同
 * @todo 可以扩展支持不同数据类型的点积计算，如double、int等
 * @code 使用示例
nvcc thrust_dot_product.cu -o thrust_dot_product
thrust_dot_product
 * @endcode
 * @result 使用示例输出结果
 *   v[0] == 1
 *   v[1] == 2
 *   v[2] == 3
 *   w[0] == 1
 *   w[1] == 1
 *   w[2] == 1
 *   dot_product(v,w)==6
 * @example 引用外部示例文件
 * @see https://docs.nvidia.com/cuda/thrust/
 * @since CUDA 3.0
 * @author mekeny1
 * @date 2025-08-13
 * @version 0.1
 * @copyright Copyright (c) 2025
 */
#include <thrust/host_vector.h>   // 主机向量头文件
#include <thrust/device_vector.h> // 设备向量头文件
#include <iostream>
using namespace std;

// 自定义函数对象 - 用于向量元素相乘
// 这是Thrust中常用的函数对象模式，支持在GPU上执行自定义操作
struct multiply_functor
{
    float w;  // 权重参数，可用于加权点积计算
    
    // 构造函数，默认权重为1
    multiply_functor(float _w = 1) : w(_w) {}

    // 重载函数调用操作符 - 在GPU上执行元素相乘
    // __device__关键字表示此函数在GPU设备上执行
    __device__ float operator()(const float &x, const float &y)
    {
        return w * x * y;  // 返回加权乘积
    }
};

// 点积计算函数 - 使用Thrust的transform-reduce模式
// 这是并行计算中的经典模式：先变换(transform)再归约(reduce)
float dot_product(thrust::device_vector<float> &v, thrust::device_vector<float> &w)
{
    // 创建临时向量存储乘积结果
    thrust::device_vector<float> z(v.size());

    // 第一步：transform - 将两个向量的对应元素相乘
    // v.begin(), v.end(): 第一个向量的迭代器范围
    // w.begin(): 第二个向量的起始迭代器
    // z.begin(): 结果向量的起始迭代器
    // multiply_functor(): 执行相乘操作的函数对象
    thrust::transform(v.begin(), v.end(), w.begin(), z.begin(), multiply_functor());

    // 第二步：reduce - 将乘积向量中的所有元素求和
    // 返回最终的标量结果（点积值）
    return thrust::reduce(z.begin(), z.end());
}

int main(void)
{
    // 创建第一个设备向量v
    thrust::device_vector<float> v;

    // 向向量v添加元素
    v.push_back(1.0f);
    v.push_back(2.0f);
    v.push_back(3.0f);

    // 创建第二个设备向量w，预分配大小为3
    thrust::device_vector<float> w(3);

    // 使用fill算法将向量w的所有元素填充为1.0
    // 这是Thrust提供的并行填充算法
    thrust::fill(w.begin(), w.end(), 1.0f);

    // 打印向量v的内容
    for (int i = 0; i < v.size(); i++)
    {
        cout << "v[" << i << "] == " << v[i] << endl;
    }

    // 打印向量w的内容
    for (int i = 0; i < w.size(); i++)
    {
        cout << "w[" << i << "] == " << w[i] << endl;
    }

    // 计算并打印点积结果
    cout << "dot_product(v,w)==" << dot_product(v, w) << endl;

    return 0;
}
