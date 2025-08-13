'''
Author: mekeny1
Date: 2025-07-02 01:15:04
LastEditors: mekeny1
LastEditTime: 2025-08-13 14:14:59
FilePath: \pycuda_tutorial_hapril\Chapter09\deep_neural_network.py
Description: 深度神经网络实现 - 基于PyCUDA的GPU加速神经网络训练
This module implements a complete deep neural network framework using PyCUDA for GPU acceleration.
It includes dense layers, softmax layers, and a sequential network architecture with backpropagation
training using finite difference gradient estimation.
    - cuda: GPU acceleration using PyCUDA
    - neural-network: Deep neural network implementation
    - backpropagation: Training algorithm using finite differences
    - iris-classification: Example application on iris dataset
    - gpu-memory: Efficient GPU memory management
    - parallel-computing: Parallel gradient computation using streams
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import numpy as np
from queue import Queue
import csv
from time import time

# 最大熵值限制，防止数值溢出
MAX_ENTROPY = 1


def cross_entropy(predictions=None, ground_truth=None):
    """
    计算交叉熵损失函数

    Args:
        predictions: 模型预测概率 (2D numpy array)
        ground_truth: 真实标签 (2D numpy array)

    Returns:
        float: 平均交叉熵损失

    Note:
        使用数值稳定性处理，限制最大熵值防止log(0)导致的无穷大
    """
    if predictions is None or ground_truth is None:
        raise Exception(
            "Error!  Both predictions and ground truth must be float32 arrays")

    p = np.array(predictions).copy()
    y = np.array(ground_truth).copy()

    if p.shape != y.shape:
        raise Exception(
            "Error!  Both predictions and ground_truth must have same shape.")

    if len(p.shape) != 2:
        raise Exception(
            "Error!  Both predictions and ground_truth must be 2D arrays.")

    total_entropy = 0.0

    # 逐元素计算交叉熵
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if y[i, j] == 1:
                # 对于正类样本，计算 -log(p)
                total_entropy += min(
                    np.abs(np.nan_to_num(np.log(p[i, j]))), MAX_ENTROPY)
            else:
                # 对于负类样本，计算 -log(1-p)
                total_entropy += min(np.abs(np.nan_to_num(np.log(1 -
                                     p[i, j]))), MAX_ENTROPY)

    return total_entropy / p.size


# CUDA核函数：密集层前向传播计算
# 包含ReLU和Sigmoid激活函数，支持权重扰动用于梯度计算
DenseEvalCode = """
#define _RELU(x) (((x) > 0.0f) ? (x) : 0.0f)  // ReLU激活函数宏定义
#define _SIGMOID(x) (1.0f / (1.0f + expf(-(x))))  // Sigmoid激活函数宏定义

__global__ void dense_eval(int num_outputs, int num_inputs, int relu, int sigmoid, 
                          float *w, float *b, float *x, float *y, int batch_size, 
                          int w_t, int b_t, float delta)
{
    // 计算当前线程对应的输出神经元索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_outputs)
    {
        // 对每个批次样本进行计算
        for (int k = 0; k < batch_size; k++)
        {
            double temp = 0.0f;  // 使用double提高数值精度

            // 计算加权和：y[i] = sum(w[i,j] * x[j]) + b[i]
            for (int j = 0; j < num_inputs; j++)
            {
                temp += ((double)w[(num_inputs)*i + j]) * ((double)x[k * num_inputs + j]);
            }
            temp += (double)b[i];
            y[k * num_outputs + i] = (float)temp;
        }

        // 权重扰动：用于有限差分梯度计算
        if (w_t >= 0 && i == (w_t / num_inputs))
        {
            int j = w_t % num_inputs;
            for (int k = 0; k < batch_size; k++)
            {
                y[k * num_outputs + i] += delta * x[k * num_inputs + j];
            }
        }

        // 偏置扰动：用于有限差分梯度计算
        if (b_t >= 0 && i == b_t)
        {
            for (int k = 0; k < batch_size; k++)
            {
                y[k * num_outputs + i] += delta;
            }
        }

        // 应用激活函数
        if (relu > 0 || sigmoid > 0)
        {
            for (int k = 0; k < batch_size; k++)
            {
                float temp = y[k * num_outputs + i];

                if (relu > 0)
                    temp = _RELU(temp);
                if (sigmoid > 0)
                    temp = _SIGMOID(temp);

                y[k * num_outputs + i] = temp;
            }
        }
    }
    return;
}
"""

# 编译CUDA核函数
eval_mod = SourceModule(DenseEvalCode)
eval_ker = eval_mod.get_function("dense_eval")


class DenserLayer:
    """
    密集层类 - 实现全连接神经网络层

    支持ReLU和Sigmoid激活函数，使用CUDA进行并行计算加速。
    包含权重扰动功能，用于有限差分梯度估计。
    """

    def __init__(self, num_inputs=None, num_outputs=None, weights=None, b=None,
                 stream=None, relu=False, sigmoid=False, delta=None):
        """
        初始化密集层

        Args:
            num_inputs: 输入特征维度
            num_outputs: 输出特征维度  
            weights: 权重矩阵 (可选，默认随机初始化)
            b: 偏置向量 (可选，默认零向量)
            stream: CUDA流对象
            relu: 是否使用ReLU激活函数
            sigmoid: 是否使用Sigmoid激活函数
            delta: 有限差分步长
        """
        self.stream = stream

        # 设置有限差分步长，用于梯度计算
        if delta is None:
            self.delta = np.float32(0.001)
        else:
            self.delta = np.float32(delta)

        # 权重初始化
        if weights is None:
            # 随机初始化权重，范围[-0.5, 0.5]
            weights = (np.random.rand(num_outputs, num_inputs)-.5)
            self.num_inputs = np.int32(num_inputs)
            self.num_outputs = np.int32(num_outputs)

        # 将权重转移到GPU内存
        if type(weights) != pycuda.gpuarray.GPUArray:
            self.weights = gpuarray.to_gpu_async(
                np.array(weights, dtype=np.float32), stream=self.stream)
        else:
            self.weights = weights

        # 设置输入输出维度
        if num_inputs is None or num_outputs is None:
            self.num_inputs = np.int32(self.weights.shape[1])
            self.num_outputs = np.int32(self.weights.shape[0])
        else:
            self.num_inputs = np.int32(num_inputs)
            self.num_outputs = np.int32(num_outputs)

        # 偏置初始化
        if b is None:
            b = gpuarray.zeros((self.num_outputs,), dtype=np.float32)

        # 将偏置转移到GPU内存
        if type(b) != pycuda.gpuarray.GPUArray:
            self.b = gpuarray.to_gpu_async(
                np.array(b, dtype=np.float32), stream=self.stream)
        else:
            self.b = b

        # 激活函数设置
        self.relu = np.int32(relu)
        self.sigmoid = np.int32(sigmoid)

        # CUDA线程块和网格配置
        self.block = (32, 1, 1)  # 每个块32个线程
        self.grid = (int(np.ceil(self.num_outputs/32)), 1, 1)  # 网格大小

    def eval_(self, x, y=None, batch_size=None, stream=None, delta=None, w_t=None, b_t=None):
        """
        前向传播计算

        Args:
            x: 输入数据 (CPU或GPU数组)
            y: 输出缓冲区 (可选)
            batch_size: 批次大小
            stream: CUDA流
            delta: 有限差分步长
            w_t: 权重扰动索引
            b_t: 偏置扰动索引

        Returns:
            GPUArray: 层输出结果
        """
        if stream is None:
            stream = self.stream

        # 确保输入数据在GPU上
        if type(x) != pycuda.gpuarray.GPUArray:
            x = gpuarray.to_gpu_async(
                np.array(x, dtype=np.float32), stream=self.stream)

        # 确定批次大小
        if batch_size is None:
            if len(x.shape) == 2:
                batch_size = np.int32(x.shape[0])
            else:
                batch_size = np.int32(1)

        # 设置有限差分参数
        if delta is None:
            delta = self.delta
        delta = np.float32(delta)

        if w_t is None:
            w_t = np.int32(-1)

        if b_t is None:
            b_t = np.int32(-1)

        # 分配输出缓冲区
        if y is None:
            if batch_size == 1:
                y = gpuarray.empty((self.num_outputs,), dtype=np.float32)
            else:
                y = gpuarray.empty(
                    (batch_size, self.num_outputs), dtype=np.float32)

        # 调用CUDA核函数进行并行计算
        eval_ker(self.num_outputs, self.num_inputs, self.relu, self.sigmoid, self.weights, self.b, x, y, np.int32(
            batch_size), w_t, b_t, delta, block=self.block, grid=self.grid, stream=stream)

        return y


# Softmax层的CUDA实现
# 分两步：1) 计算指数 2) 归一化

# 第一步：计算指数
SoftmaxExpCode = """
__global__ void softmax_exp(int num, float *x, float *y, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num)
    {
        // 对每个批次样本计算指数
        for (int k = 0; k < batch_size; k++)
        {
            y[num * k + i] = expf(x[num * k + i]);
        }
    }
}
"""
exp_mod = SourceModule(SoftmaxExpCode)
exp_ker = exp_mod.get_function("softmax_exp")

# 第二步：归一化计算
SoftmaxMeanCode = """
__global__ void softmax_mean(int num, float *x, float *y, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size)
    {
        float temp = 0.0f;

        // 计算分母：所有指数项的和
        for (int k = 0; k < num; k++)
            temp += x[num * i + k];

        // 归一化：每个指数项除以总和
        for (int k = 0; k < num; k++)
            y[i * num + k] = x[i * num + k] / temp;
    }
    return;
}
"""
mean_mod = SourceModule(SoftmaxMeanCode)
mean_ker = mean_mod.get_function("softmax_mean")


class SoftmaxLayer:
    """
    Softmax层类 - 实现多分类概率输出

    将神经网络的原始输出转换为概率分布，用于多分类问题。
    使用CUDA并行计算指数和归一化操作。
    """

    def __init__(self, num=None, stream=None):
        """
        初始化Softmax层

        Args:
            num: 类别数量
            stream: CUDA流对象
        """
        self.num = np.int32(num)
        self.stream = stream

    def eval_(self, x, y=None, batch_size=None, stream=None):
        """
        Softmax前向传播计算

        Args:
            x: 输入数据 (原始logits)
            y: 输出缓冲区 (可选)
            batch_size: 批次大小
            stream: CUDA流

        Returns:
            GPUArray: 概率分布输出
        """
        if stream is None:
            stream = self.stream

        # 确保输入数据在GPU上
        if type(x) != pycuda.gpuarray.GPUArray:
            temp = np.array(x, dtype=np.float32)
            x = gpuarray.to_gpu_async(temp, stream=stream)

        # 确定批次大小
        if batch_size == None:
            if len(x.shape) == 2:
                batch_size = np.int32(x.shape[0])
            else:
                batch_size = np.int32(1)
        else:
            batch_size = np.int32(batch_size)

        # 分配输出缓冲区
        if y is None:
            if batch_size == 1:
                y = gpuarray.empty((self.num,), dtype=np.float32)
            else:
                y = gpuarray.empty((batch_size, self.num), dtype=np.float32)

        # 第一步：计算指数
        exp_ker(self.num, x, y, batch_size, block=(32, 1, 1), grid=(
            int(np.ceil(batch_size/32)), 1, 1), stream=stream)

        # 第二步：归一化
        mean_ker(self.num, y, y, batch_size, block=(32, 1, 1), grid=(
            int(np.ceil(batch_size/32)), 1, 1), stream=stream)

        return y


class SequentialNetwork:
    """
    序列神经网络类 - 实现多层神经网络架构

    支持任意层数的神经网络，使用有限差分方法进行梯度计算。
    利用CUDA流实现并行梯度计算，提高训练效率。
    """

    def __init__(self, layers=None, delta=None, stream=None, max_batch_size=32, max_streams=10, epochs=10):
        """
        初始化序列神经网络

        Args:
            layers: 层配置列表
            delta: 有限差分步长
            stream: 主CUDA流
            max_batch_size: 最大批次大小
            max_streams: 最大CUDA流数量
            epochs: 训练轮数
        """
        self.network = []  # 网络层列表
        self.network_summary = []  # 网络结构摘要
        self.network_mem = []  # GPU内存缓冲区

        # 创建CUDA流
        if stream is not None:
            self.stream = stream
        else:
            self.stream = drv.Stream()

        # 设置有限差分步长
        if delta is None:
            delta = 0.0001

        self.delta = delta
        self.max_batch_size = max_batch_size
        self.max_streams = max_streams
        self.epochs = epochs

        # 添加预定义的层
        if layers is not None:
            for layer in layers:
                self.add_layer(self, layer)

    def add_layer(self, layer):
        """
        添加网络层

        Args:
            layer: 层配置字典，包含类型、参数等信息
        """
        if layer["type"] == "dense":
            # 确定输入维度
            if len(self.network) == 0:
                num_inputs = layer["num_inputs"]
            else:
                num_inputs = self.network_summary[-1][2]

            num_outputs = layer["num_outputs"]
            sigmoid = layer["sigmoid"]
            relu = layer["relu"]
            weights = layer["weights"]
            b = layer["bias"]

            # 创建密集层
            self.network.append(DenserLayer(
                num_inputs=num_inputs, num_outputs=num_outputs, sigmoid=sigmoid, relu=relu, weights=weights, b=b))
            self.network_summary.append(("dense", num_inputs, num_outputs))

            # 为批次处理分配GPU内存
            if self.max_batch_size > 1:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty(
                        (self.max_batch_size, self.network_summary[-1][1]), dtype=np.float32))
                self.network_mem.append(gpuarray.empty(
                    (self.max_batch_size, self.network_summary[-1][2]), dtype=np.float32))
            else:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty(
                        (self.network_summary[-1][1],), dtype=np.float32))
                self.network_mem.append(gpuarray.empty(
                    (self.network_summary[-1][2],), dtype=np.float32))

        elif layer["type"] == "softmax":
            # 验证softmax层前必须有密集层
            if len(self.network) == 0:
                raise Exception(
                    "Error!  Need a dense layer before a softmax layer")

            if self.network_summary[-1][0] != "dense":
                raise Exception(
                    "Error! Need a dense layer before a softmax layer!")

            num = self.network_summary[-1][2]

            # 创建softmax层
            self.network.append(SoftmaxLayer(num=num))
            self.network_summary.append(("softmax", num, num))

            # 分配GPU内存
            if self.max_batch_size > 1:
                self.network_mem.append(gpuarray.empty((
                    self.max_batch_size, self.network_summary[-1][2]), dtype=np.float32))
            else:
                self.network_mem.append(gpuarray.empty((
                    self.network_summary[-1][2],), dtype=np.float32))

    def predict(self, x, stream=None):
        """
        网络前向传播预测

        Args:
            x: 输入数据
            stream: CUDA流

        Returns:
            numpy.ndarray: 预测结果
        """
        if stream is None:
            stream = self.stream

        # 确保输入是numpy数组
        if type(x) != np.ndarray:
            temp = np.array(x, dtype=np.float32)
            x = temp

        # 将输入数据复制到GPU内存缓冲区
        if (x.size == self.network_mem[0].size):
            self.network_mem[0].set_async(x, stream=stream)
        else:
            if x.size > self.network_mem[0].size:
                raise Exception("Error: batch size too large for input.")

            # 填充零值以适应缓冲区大小
            x0 = np.zeros((self.network_mem[0].size,), dtype=np.float32)
            x0[0:x.size] = x.ravel()
            self.network_mem[0].set_async(x0.reshape(
                self.network_mem[0].shape), stream=stream)

        # 确定批次大小
        if len(x.shape) == 2:
            batch_size = x.shape[0]
        else:
            batch_size = 1

        # 逐层前向传播
        for i in range(len(self.network)):
            self.network[i].eval_(
                x=self.network_mem[i], y=self.network_mem[i+1], batch_size=batch_size, stream=stream)

        # 获取最终输出
        y = self.network_mem[-1].get_async(stream=stream)

        # 裁剪到实际批次大小
        if len(y.shape) == 2:
            y = y[0:batch_size, :]

        return y

    def partial_predict(self, layer_index=None, w_t=None, b_t=None, partial_mem=None, stream=None, batch_size=None, delta=None):
        """
        部分前向传播 - 用于梯度计算

        从指定层开始前向传播，支持权重和偏置扰动

        Args:
            layer_index: 起始层索引
            w_t: 权重扰动索引
            b_t: 偏置扰动索引
            partial_mem: 部分内存缓冲区
            stream: CUDA流
            batch_size: 批次大小
            delta: 扰动步长
        """
        # 从指定层开始计算，应用扰动
        self.network[layer_index].eval_(x=self.network_mem[layer_index], y=partial_mem[layer_index+1],
                                        batch_size=batch_size, stream=stream, w_t=w_t, b_t=b_t, delta=delta)

        # 继续后续层的前向传播
        for i in range(layer_index+1, len(self.network)):
            self.network[i].eval_(
                x=partial_mem[i], y=partial_mem[i+1], batch_size=batch_size, stream=stream)

    def bsgd(self, training=None, labels=None, delta=None, max_streams=None, batch_size=None, epochs=1, training_rate=0.01):
        """
        批量随机梯度下降训练

        使用有限差分方法计算梯度，通过多个CUDA流并行计算提高效率。

        Args:
            training: 训练数据
            labels: 训练标签
            delta: 有限差分步长
            max_streams: 最大CUDA流数量
            batch_size: 批次大小
            epochs: 训练轮数
            training_rate: 学习率
        """
        training_rate = np.float32(training_rate)

        # 确保数据类型正确
        training = np.float32(training)
        labels = np.float32(labels)

        if training.shape[0] != labels.shape[0]:
            raise Exception(
                "Number of training data points should be same as labels!")

        # 设置默认参数
        if max_streams is None:
            max_streams = self.max_streams

        if epochs is None:
            epochs = self.epochs

        if delta is None:
            delta = self.delta

        # 创建多个CUDA流和对应的内存缓冲区
        streams = []
        bgd_mem = []

        for _ in range(max_streams):
            streams.append(drv.Stream())
            bgd_mem.append([])

        # 为每个流分配GPU内存缓冲区
        for i in range(len(bgd_mem)):
            for mem_bank in self.network_mem:
                bgd_mem[i].append(gpuarray.empty_like(mem_bank))

        num_points = training.shape[0]

        if batch_size is None:
            batch_size = self.max_batch_size

        # 创建索引列表用于随机打乱
        index = list(range(training.shape[0]))

        # 开始训练循环
        for k in range(epochs):
            print("-----------------------------------------------------------")
            print("Starting training epoch: %s" % k)
            print("Batch size: %s , Total number of training samples: %s" %
                  (batch_size, num_points))
            print("-----------------------------------------------------------")

            all_grad = []

            # 随机打乱训练数据
            np.random.shuffle(index)

            # 批次训练
            for r in range(int(np.floor(training.shape[0]/batch_size))):
                batch_index = index[r*batch_size:(r+1)*batch_size]

                batch_training = training[batch_index, :]
                batch_labels = labels[batch_index, :]

                # 前向传播
                batch_predictions = self.predict(batch_training)

                # 计算当前损失
                cur_entropy = cross_entropy(
                    predictions=batch_predictions, ground_truth=batch_labels)

                print("entropy: %s" % cur_entropy)

                # 对每个密集层计算梯度
                for i in range(len(self.network)):
                    if self.network_summary[i][0] != "dense":
                        continue

                    # 创建权重和偏置队列
                    all_weights = Queue()

                    # 初始化梯度数组
                    grad_w = np.zeros(
                        (self.network[i].weights.size,), dtype=np.float32)
                    grad_b = np.zeros(
                        (self.network[i].b.size,), dtype=np.float32)

                    # 将所有权重和偏置加入队列
                    for w in range(self.network[i].weights.size):
                        all_weights.put(("w", np.int32(w)))

                    for b in range(self.network[i].b.size):
                        all_weights.put(("b", np.int32(b)))

                    # 使用多个流并行计算梯度
                    while not all_weights.empty():
                        streams_weights = Queue()

                        # 为每个流分配计算任务
                        for j in range(max_streams):
                            if all_weights.empty():
                                break

                            wb = all_weights.get()

                            if wb[0] == "w":
                                w_t = wb[1]
                                b_t = None
                            elif wb[0] == "b":
                                b_t = wb[1]
                                w_t = None

                            streams_weights.put(wb)

                            # 在对应流上执行部分前向传播
                            self.partial_predict(
                                layer_index=i, w_t=w_t, b_t=b_t, partial_mem=bgd_mem[j], stream=streams[j], batch_size=batch_size, delta=delta)

                        # 收集所有流的结果并计算梯度
                        for j in range(max_streams):
                            if streams_weights.empty():
                                break

                            wb = streams_weights.get()

                            # 获取扰动后的预测结果
                            w_predictions = bgd_mem[j][-1].get_async(
                                stream=streams[j])

                            # 计算扰动后的损失
                            w_entropy = cross_entropy(
                                predictions=w_predictions[:batch_size, :], ground_truth=batch_labels)

                            # 使用有限差分计算梯度
                            if wb[0] == "w":
                                w_t = wb[1]
                                grad_w[w_t] = -(w_entropy-cur_entropy)/delta
                            elif wb[0] == "b":
                                b_t = wb[1]
                                grad_b[b_t] = -(w_entropy-cur_entropy)/delta

                    # 保存当前层的梯度
                    all_grad.append(
                        [np.reshape(grad_w, self.network[i].weights.shape), grad_b])

            # 更新网络参数
            for i in range(len(self.network)):
                if self.network_summary[i][0] == "dense":
                    # 获取当前参数
                    new_weights = self.network[i].weights.get()
                    new_weights += training_rate*all_grad[i][0]  # 权重更新
                    new_bias = self.network[i].b.get()
                    new_bias += training_rate*all_grad[i][1]     # 偏置更新

                    # 将更新后的参数设置回GPU
                    self.network[i].weights.set(new_weights)
                    self.network[i].b.set(new_bias)


def condition_data(data, means=None, stds=None):
    """
    数据标准化处理

    对输入数据进行标准化，使每个特征的均值为0，标准差为1。

    Args:
        data: 输入数据
        means: 预计算的均值 (可选)
        stds: 预计算的标准差 (可选)

    Returns:
        tuple: (标准化后的数据, 均值, 标准差)
    """
    if means is None:
        means = np.mean(data, axis=0)

    if stds is None:
        stds = np.std(data, axis=0)

    conditioned_data = data.copy()
    conditioned_data -= means
    conditioned_data /= stds

    return (conditioned_data, means, stds)


if __name__ == "__main__":
    # 鸢尾花数据集分类示例

    # 类别编码映射
    to_class = {"Iris-setosa": [1, 0, 0],
                "Iris-versicolor": [0, 1, 0], "Iris-virginica": [0, 0, 1]}

    iris_data = []
    iris_labels = []

    # 读取鸢尾花数据集
    with open("./iris.data", "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            newrow = []
            if len(row) != 5:
                break
            for i in range(4):
                newrow.append(row[i])
            iris_data.append(newrow)
            iris_labels.append(to_class[row[4]])

    # 数据预处理
    iris_len = len(iris_data)
    shuffled_index = list(range(iris_len))
    np.random.shuffle(shuffled_index)  # 随机打乱数据
    iris_data = np.float32(iris_data)
    iris_labels = np.float32(iris_labels)
    iris_data = iris_data[shuffled_index, :]
    iris_labels = iris_labels[shuffled_index, :]

    # 划分训练集和测试集 (2:1比例)
    t_len = (2*iris_len)//3

    iris_train = iris_data[:t_len, :]
    label_train = iris_labels[:t_len, :]

    iris_test = iris_data[t_len:, :]
    label_test = iris_labels[t_len:, :]

    # 创建神经网络
    sn = SequentialNetwork(max_batch_size=32)

    # 构建网络架构：4输入 -> 10 -> 15 -> 20 -> 3输出
    sn.add_layer({"type": "dense", "num_inputs": 4, "num_outputs": 10,
                 "relu": True, "sigmoid": False, "weights": None, "bias": None})
    sn.add_layer({"type": "dense", "num_inputs": 10, "num_outputs": 15,
                 "relu": True, "sigmoid": False, "weights": None, "bias": None})
    sn.add_layer({"type": "dense", "num_inputs": 15, "num_outputs": 20,
                 "relu": True, "sigmoid": False, "weights": None, "bias": None})
    sn.add_layer({"type": "dense", "num_inputs": 20, "num_outputs": 3,
                 "relu": True, "sigmoid": False, "weights": None, "bias": None})
    sn.add_layer({"type": "softmax"})

    # 数据标准化
    ctrain, means, stds = condition_data(iris_train)

    # 训练网络
    t1 = time()
    sn.bsgd(training=ctrain, labels=label_train, batch_size=16,
            max_streams=20, epochs=10, delta=0.001, training_rate=0.01)
    training_time = time()-t1

    # 测试网络性能
    hits = 0
    ctest, _, _ = condition_data(iris_test, means=means, stds=stds)
    for i in range(ctest.shape[0]):
        if np.argmax(sn.predict(ctest[i, :])) == np.argmax(label_test[i, :]):
            hits += 1

    # 输出结果
    print("Percentage Correct Classifications: %s" %
          (float(hits)/ctest.shape[0]))
    print("Total Training Time: %s" % training_time)
