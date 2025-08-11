'''
Author: mekeny1
Date: 2025-05-28 01:00:21
LastEditors: mekeny1
LastEditTime: 2025-08-11 01:18:37
FilePath: \pycuda_tutorial_hapril\Chapter03\deviceQuery.py
Description: 
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import pycuda.driver as drv

# 初始化CUDA驱动，建立与GPU的通信连接
# 这是使用PyCUDA进行GPU编程的第一步
drv.init()

"""
CUDA设备查询程序

该程序用于检测和显示系统中所有CUDA GPU设备的详细信息。
类似于NVIDIA官方工具nvidia-smi，但提供更详细的硬件规格信息。

算法处理流程：
1. 初始化CUDA驱动环境
2. 检测系统中可用的CUDA设备数量
3. 遍历每个设备，获取其详细属性
4. 计算CUDA核心数量（基于计算能力和多处理器数量）
5. 显示设备的完整规格信息

核心方法：
- drv.Device.count(): 获取系统中CUDA设备总数
- drv.Device(i): 创建第i个设备的对象
- device.get_attributes(): 获取设备的所有属性
- device.compute_capability(): 获取计算能力版本

CUDA相关概念：
- 计算能力（Compute Capability）：GPU架构版本，决定支持的特性和性能
- 多处理器（Multiprocessor）：GPU中的计算单元，每个包含多个CUDA核心
- CUDA核心：GPU中的基本计算单元，执行浮点和整数运算
- 设备属性：GPU的详细硬件规格，如内存大小、线程块限制等

软硬件特性：
- GPU架构：不同计算能力对应不同的GPU架构（如Maxwell、Pascal、Volta等）
- 内存层次：全局内存、共享内存、寄存器等不同级别的存储
- 并行计算：SIMT（单指令多线程）执行模型
- 硬件调度：GPU自动管理线程调度和资源分配
"""

# 检测并显示系统中CUDA设备的数量
print("检测到 {} 个CUDA设备".format(drv.Device.count()))

# 遍历每个CUDA设备，获取详细信息
for i in range(drv.Device.count()):
    # 创建第i个GPU设备的对象
    gpu_device = drv.Device(i)

    # 显示设备基本信息
    print("设备 {}: {}".format(i, gpu_device.name()))

    # 获取计算能力版本（如8.6、7.5等）
    # 计算能力决定了GPU支持的特性和性能上限
    compute_capability = float("%d.%d" % gpu_device.compute_capability())
    print("\t 计算能力: {}".format(compute_capability))

    # 获取设备总内存大小（以MB为单位）
    # GPU内存是存储数据和代码的主要空间
    print("\t 总内存: {} MB".format(
        gpu_device.total_memory()//(1024**2)))

    # 获取设备的所有属性
    # 这些属性包含了GPU的详细硬件规格
    device_attributes_tuples = gpu_device.get_attributes().items()
    device_attributes = {}

    # 将属性转换为字典格式，便于处理
    for k, v in device_attributes_tuples:
        device_attributes[str(k)] = v

    # 获取多处理器数量
    # 多处理器是GPU中的主要计算单元
    num_mp = device_attributes["MULTIPROCESSOR_COUNT"]

    # 根据计算能力确定每个多处理器的CUDA核心数量
    # 不同架构的GPU每个多处理器包含的CUDA核心数不同
    cuda_cores_per_mp = {5.0: 128, 5.1: 128, 5.2: 128,
                         6.0: 64, 6.1: 128, 6.2: 128, 8.6: 16}[compute_capability]

    # 计算并显示总CUDA核心数量
    # 总核心数 = 多处理器数量 × 每个多处理器的核心数
    print("\t ({}) 个多处理器, ({}) CUDA核心/多处理器: {} 个CUDA核心".format(
        num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp))

    # 从属性字典中移除多处理器数量，避免重复显示
    device_attributes.pop("MULTIPROCESSOR_COUNT")

    # 显示设备的其他所有属性
    # 这些属性包括线程块限制、内存配置、缓存大小等
    for k in device_attributes.keys():
        print("\t {}: {}".format(k, device_attributes[k]))
