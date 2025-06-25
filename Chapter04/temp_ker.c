__global__ void up_ker(double *x, double *old_x, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int _2k = 1 << k;
    int _2k1 = 1 << (k + 1);

    int j = tid * _2k1;

    x[j + _2k1 - 1] = old_x[j + _2k - 1] + old_x[j + _2k1 - 1];
}