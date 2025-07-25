import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda import gpuarray
import numpy as np
from matplotlib import pyplot as plt
from time import time
import matplotlib
matplotlib.use("agg")


mandel_ker = ElementwiseKernel(
    "pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
    """
    mandelbrot_graph[i]=1;
    pycuda::complex<float> c=lattice[i];
    pycuda::complex<float> z(0,0);
    for (int j=0; j<max_iters; j++) 
    {
        z=z*z+c;
        if (abs(z) > upper_bound) 
        {
            mandelbrot_graph[i]=0;
            break;
        }
    }
    """,
    "mandelbrot_ker"
)


def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    real_vals = np.matrix(np.linspace(
        real_low, real_high, width), dtype=np.complex64)
    image_vals = np.matrix(np.linspace(
        imag_low, imag_high, height), dtype=np.complex64)*1j
    mandelbrot_lattice = np.array(
        real_vals+image_vals.transpose(), dtype=np.complex64)

    # 异步复制到gpu
    mandelbrot_lattice_gpu = gpuarray.to_gpu_async(mandelbrot_lattice)
    pycuda.autoinit.context.synchronize()

    mandelbrot_graph_gpu = gpuarray.empty(
        shape=mandelbrot_lattice.shape, dtype=np.float32)

    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu,
               np.int32(max_iters), np.float32(upper_bound))
    pycuda.autoinit.context.synchronize()

    mandelbrot_graph = mandelbrot_graph_gpu.get_async()
    pycuda.autoinit.context.synchronize()

    return mandelbrot_graph


if __name__ == "__main__":
    t1 = time()
    mandel = gpu_mandelbrot(512, 512, -2, 2, -2, 2, 256, 2)
    t2 = time()

    mandel_time = t2 - t1

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)
    t2 = time()

    dump_time = t2 - t1

    print('It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time))
    print('It took {} seconds to dump the image.'.format(dump_time))
