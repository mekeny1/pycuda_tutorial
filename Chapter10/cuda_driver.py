'''
Author: mekeny1
Date: 2025-07-10 15:10:13
LastEditors: mekeny1
LastEditTime: 2025-08-13 13:26:48
FilePath: \pycuda_tutorial_hapril\Chapter10\cuda_driver.py
Description: 
CUDA Driver API Python封装模块
@overview: 提供CUDA Driver API的Python接口封装，支持跨平台CUDA编程
@architecture: 使用ctypes库实现C/C++ CUDA Driver API的Python绑定
@platform: 支持Windows (nvcuda.dll) 和 Linux (libcuda.so) 平台
@features: 设备管理、上下文创建、内存操作、内核启动等核心CUDA功能
@usage: 用于底层CUDA编程，提供比PyCUDA更直接的Driver API访问
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from ctypes import *
import sys

# 跨平台CUDA库加载
# 根据操作系统自动选择对应的CUDA动态库
# 使用ctypes.CDLL加载CUDA Driver API库
if "linux" in sys.platform:
    cuda = CDLL("libcuda.so")  # Linux平台CUDA库
elif "win" in sys.platform:
    cuda = CDLL("nvcuda.dll")  # Windows平台CUDA库

# CUDA错误码映射表
# 提供CUDA错误码到错误描述的映射
# 便于调试时快速识别CUDA操作错误类型
CUDA_ERROS = {0: "CUDA_SUCCESS", 1: "CUDA_ERROR_INVALID_VALUE", 200: "CUDA_ERROR_INVALID_IMAGE",
              201: "CUDA_ERROR_INVALID_CONTEXT", 400: "CUAD_ERROR_INVALID_HANDLE"}

# ==================== CUDA Driver API 函数封装 ====================

# 初始化CUDA Driver API
# 初始化CUDA Driver API，必须在其他CUDA操作前调用
# 指定初始化标志，通常为0
cuInit = cuda.cuInit
cuInit.argtypes = [c_uint]  # 参数类型：无符号整数
cuInit.restype = int        # 返回值：CUDA错误码

# 获取系统中NVIDIA GPU设备数量
# 枚举系统中可用的CUDA设备
# 输出参数，存储设备数量
cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]  # 参数类型：整数指针
cuDeviceGetCount.restype = int                # 返回值：CUDA错误码

# 获取指定索引的CUDA设备句柄
# 获取特定设备的句柄用于后续操作
# 基于设备索引获取设备句柄
cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]  # 参数类型：设备句柄指针，设备索引
cuDeviceGet.restype = int                       # 返回值：CUDA错误码

# 创建CUDA上下文（绑定到指定设备）
# 创建CUDA执行上下文，管理GPU资源
# 将上下文绑定到特定GPU设备
# 上下文创建标志，控制上下文行为
cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [c_void_p, c_uint, c_int]  # 参数类型：上下文指针，标志，设备句柄
cuCtxCreate.restype = int                         # 返回值：CUDA错误码

# 加载PTX（Parallel Thread Execution）模块文件
# 从文件系统加载编译好的PTX代码
# 使PTX中的内核函数可用于执行
cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [c_void_p, c_char_p]  # 参数类型：模块句柄指针，PTX文件路径
cuModuleLoad.restype = int                    # 返回值：CUDA错误码

# 同步CUDA上下文中的所有操作
# 等待所有GPU操作完成
# 阻塞调用线程直到GPU操作完成
cuCtxSynchronize = cuda.cuCtxSynchronize
cuCtxSynchronize.argtypes = []  # 无参数
cuCtxSynchronize.restype = int  # 返回值：CUDA错误码

# 从加载的模块中获取内核函数句柄
# 根据函数名获取内核函数句柄
# 为内核启动准备函数句柄
cuModuleGetFunction = cuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [
    c_void_p, c_void_p, c_char_p]  # 参数类型：函数句柄指针，模块句柄，函数名
cuModuleGetFunction.restype = int                              # 返回值：CUDA错误码

# ==================== 内存管理函数 ====================

# 在GPU上分配全局内存
# 在GPU设备内存中分配指定大小的内存块
# 分配全局内存，所有线程块可访问
cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [c_void_p, c_size_t]  # 参数类型：内存指针，分配大小（字节）
cuMemAlloc.restype = int                    # 返回值：CUDA错误码

# 从主机内存复制数据到GPU设备内存
# 数据传输方向：CPU内存 → GPU内存
# 异步内存传输操作
cuMemcpyHtoD = cuda.cuMemcpyHtoD
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p,
                         c_size_t]  # 参数类型：GPU内存地址，CPU内存地址，传输大小
cuMemAlloc.restype = int                                 # 返回值：CUDA错误码

# 从GPU设备内存复制数据到主机内存
# 数据传输方向：GPU内存 → CPU内存
# 获取GPU计算结果到CPU内存
cuMemcpyDtoH = cuda.cuMemcpyDtoH
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p,
                         c_size_t]  # 参数类型：CPU内存地址，GPU内存地址，传输大小
cuMemcpyDtoH.restype = int                               # 返回值：CUDA错误码

# 释放GPU设备内存
# 释放之前分配的GPU内存
# 防止GPU内存泄漏
cuMemFree = cuda.cuMemFree
cuMemFree.argtypes = [c_void_p]  # 参数类型：GPU内存地址
cuMemFree.restype = int          # 返回值：CUDA错误码

# ==================== 内核执行函数 ====================

# 启动CUDA内核函数
# 在GPU上执行并行内核函数
# 配置线程网格和线程块结构
# 传递内核函数参数
cuLaunchKernel = cuda.cuLaunchKernel
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint,
                           c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]
# 参数类型：函数句柄, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
#         共享内存大小, 参数结构体, 额外参数, 流句柄
cuLaunchKernel.restype = int  # 返回值：CUDA错误码

# ==================== 上下文管理函数 ====================

# 销毁CUDA上下文
# 释放CUDA上下文占用的所有资源
# 确保GPU资源正确释放
cuCtxDestroy = cuda.cuCtxDestroy
cuCtxDestroy.argtypes = [c_void_p]  # 参数类型：上下文句柄
cuCtxDestroy.restype = int          # 返回值：CUDA错误码
