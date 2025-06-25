from __future__ import division
import pycuda.autoinit
from pycuda import gpuarray

import numpy as np
from skcuda import fft, linalg
from matplotlib import pyplot as plt


def cufft_conv(x, y):
    x = x.astype(np.complex64)
    y = y.astype(np.complex64)

    if (x.shape != y.shape):
        return -1

    plan = fft.Plan(x.shape, np.complex64, np.complex64)
    inverse_plan = fft.Plan(x.shape, np.complex64, np.complex64)

    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)

    x_fft = gpuarray.empty_like(x_gpu, dtype=np.complex64)
    y_fft = gpuarray.empty_like(y_gpu, dtype=np.complex64)
    out_gpu = gpuarray.empty_like(x_gpu, dtype=np.complex64)

    fft.fft(x_gpu, x_fft, plan)
    fft.fft(y_gpu, y_fft, plan)

    linalg.multiply(x_fft, y_fft, overwrite=True)

    fft.ifft(y_fft, out_gpu, inverse_plan, scale=True)
    conv_out = out_gpu.get()

    return conv_out


def conv_2d(ker, img):
    paddled_ker = np.zeros(
        (img.shape[0]+2*ker.shape[0], img.shape[1]+2*ker.shape[1])).astype(np.float32)
    paddled_ker[:ker.shape[0], :ker.shape[1]] = ker

    # paddled_ker 的四个角分别是高斯核被分割后的部分，每个部分近似为 15 * 15 大小（忽略中心元素）
    paddled_ker = np.roll(paddled_ker, shift=-ker.shape[0]//2, axis=0)
    paddled_ker = np.roll(paddled_ker, shift=-ker.shape[1]//2, axis=1)

    paddled_img = np.zeros_like(paddled_ker).astype(np.float32)
    # paddled_img 中间放置了要处理的图像，图像距离四条边的距离都是 31
    paddled_img[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]] = img

    out_ = cufft_conv(paddled_ker, paddled_img)

    output = out_[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]]

    return output


def gaussian_filter(x, y, sigma):
    return (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2*(sigma**2)))


def gaussian_ker(sigma):
    ker_ = np.zeros((2*sigma+1, 2*sigma+1))
    for i in range(2*sigma+1):
        for j in range(2*sigma+1):
            ker_[i, j] = gaussian_filter(i-sigma, j-sigma, sigma)
    total_ = np.sum(ker_.ravel())

    ker_ = ker_/total_

    return ker_


if __name__ == "__main__":
    rei = np.float32(plt.imread("rei.jpg"))/255
    rei_blurred = np.zeros_like(rei)

    ker = gaussian_ker(15)

    for k in range(3):
        rei_blurred[:, :, k] = conv_2d(ker, rei[:, :, k])

    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle("Guassian Filtering", fontsize=20)
    ax0.set_title("Before")
    ax0.axis("off")
    ax0.imshow(rei)
    ax1.set_title("After")
    ax1.axis("off")
    ax1.imshow(rei_blurred)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
