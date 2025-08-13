/**
 * @file thrust_demo.cu
 * @brief Thrust库演示 - CUDA并行算法库基础用法示例
 * @details 本文件演示了Thrust库的基本使用方法，包括主机向量(host_vector)和
 *          设备向量(device_vector)的创建、操作和数据传输。Thrust是CUDA的
 *          并行算法库，提供了类似STL的接口，简化了GPU编程。
 * @note Thrust库提供了丰富的并行算法，如排序、归约、扫描等，大大简化了
 *       GPU并行编程的复杂度。本示例展示了最基本的向量操作。
 * @warning 需要确保CUDA Toolkit和Thrust库已正确安装
 * @todo 可以扩展演示更多Thrust算法，如sort、reduce、transform等
 * @code 使用示例
 *   // 编译命令
 *   nvcc -o thrust_demo thrust_demo.cu
 *   
 *   // 运行命令
 *   ./thrust_demo
 * @endcode
 * @result 使用示例输出结果
 *   v[0] == 1
 *   v[1] == 2
 *   v[2] == 3
 *   v[3] == 4
 *   v_gpu[0] == 1
 *   v_gpu[1] == 2
 *   v_gpu[2] == 3
 *   v_gpu[3] == 4
 *   v_gpu[4] == 5
 * @example 引用外部示例文件
 * @see https://docs.nvidia.com/cuda/thrust/
 * @since CUDA 3.0
 * @author mekeny1
 * @date 2025-08-13
 * @version 0.1
 * @copyright Copyright (c) 2025
 */
#include <thrust/host_vector.h>   // 主机向量头文件 - 在CPU内存中存储数据
#include <thrust/device_vector.h> // 设备向量头文件 - 在GPU内存中存储数据
#include <iostream>
using namespace std;

int main()
{
    // 创建主机向量 - 在CPU内存中分配和操作数据
    thrust::host_vector<int> v;
    
    // 向主机向量添加元素 - 类似STL vector的操作
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);

    // 遍历并打印主机向量中的元素
    for (int i = 0; i < v.size(); i++)
    {
        cout << "v[" << i << "] == " << v[i] << endl;
    }

    // 将主机向量复制到设备向量 - 自动数据传输到GPU内存
    // 这是Thrust库的重要特性：自动内存管理
    thrust::device_vector<int> v_gpu = v;

    // 在GPU上向设备向量添加新元素
    // 这个操作在GPU内存中执行，无需手动内存管理
    v_gpu.push_back(5);

    // 遍历并打印设备向量中的元素
    // 访问设备向量时会自动将数据传输回CPU进行显示
    for (int i = 0; i < v_gpu.size(); i++)
    {
        std::cout << "v_gpu[" << i << "] == " << v_gpu[i] << std::endl;
    }

    return 0;
}
