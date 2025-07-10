from ctypes import *
import sys
if "linux" in sys.platform:

    cuda = CDLL("libcuda.so")
elif "win" in sys.platform:
    cuda = CDLL("nvcuda.dll")

CUDA_ERROS = {0: "CUDA_SUCCESS", 1: "CUDA_ERROR_INVALID_VALUE", 200: "CUDA_ERROR_INVALID_IMAGE",
              201: "CUDA_ERROR_INVALID_CONTEXT", 400: "CUAD_ERROR_INVALID_HANDLE"}

# 初始化Driver API
cuInit = cuda.cuInit
cuInit.argtypes = [c_uint]
cuInit.restype = int

# NVIDIA GPU 设备数
cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]
cuDeviceGetCount.restype = int

# 获取设备
cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet.restype = int

# 创建上下文（使用设备句柄）
cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [c_void_p, c_uint, c_int]
cuCtxCreate.restype = int

# 加载PTX文件
cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [c_void_p, c_char_p]
cuModuleLoad.restype = int

# 在当前CUDA上下文上同步所有启动的操作
cuCtxSynchronize = cuda.cuCtxSynchronize
cuCtxSynchronize.argtypes = []
cuCtxSynchronize.restype = int

# 从加载的模块中检索内核函数的句柄
cuModuleGetFunction = cuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p]
cuModuleGetFunction.restype = int

# 标准动态内存函数包装器
cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyHtoD = cuda.cuMemcpyHtoD
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyDtoH = cuda.cuMemcpyDtoH
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyDtoH.restype = int

cuMemFree = cuda.cuMemFree
cuMemFree.argtypes = [c_void_p]
cuMemFree.restype = int

# 为cuLaunchKernel函数编写一个包装器
cuLaunchKernel = cuda.cuLaunchKernel
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint,
                           c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]
cuLaunchKernel.restype = int

# 销毁GPU上的上下文
cuCtxDestroy = cuda.cuCtxDestroy
cuCtxDestroy.argtypes = [c_void_p]
cuCtxDestroy.restype = int
