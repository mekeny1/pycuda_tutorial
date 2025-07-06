__global__ void softmax_exp(int num, float *x, float *y, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num)
    {
        for (int k = 0; k < batch_size; k++)
        {
            y[num * k + i] = expf(x[num * k + i]);
        }
    }
}

__global__ void softmax_mean(int num, float *x, float *y, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size)
    {
        float temp = 0.0f;

        for (int k = 0; k < num; k++)
            temp += x[num * i + k];

        for (int k = 0; k < num; k++)
            y[i * num + k] = x[i * num + k] / temp;
    }
    return;
}