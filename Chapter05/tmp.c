__global__ void mult_ker(float *array, int array_len)
{
    int thd = blockIdx.x * blockDim.x + threadIdx.x;
    int num_iters = array_len / blockDim.x;
    for (int j = 0; j < num_iters; j++)
    {
        int i = j * blockDim.x + thd;
        for (int k = 0; k < 50; k++)
        {
            array[i] *= 2.0;
            array[i] /= 2.0;
        }
    }
}