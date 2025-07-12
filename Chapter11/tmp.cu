__device__ int partition(int *a, int *lo, int hi)
{
    int i = lo;
    int pivot = a[hi];
    int temp;

    for (int k = lo; k < hi; k++)
    {
        if (a[k] < pivot)
        {
            temp = a[k];
            a[k] = a[i];
            a[i] = temp;
            i++;
        }
    }

    a[hi] = a[i];
    a[i] = pivot;
    return i;
}

__global__ void quicksort_ker(int *a, int lo, int hi)
{
    cudaStream_t s_left, s_right;
    cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_right, cudaStreamNonBlocking);

    int mid = partition(a, lo, hi);
    if (mid - 1 - lo > 0)
        quicksort_ker<<<1, 1, 0, s_left>>>(a, lo, mid - 1);
    if (hi - (mid + 1) > 0)
        quicksort_ker<<<1, 1, 0, s_right>>>(a, mid + 1, hi);
    cudaStreamDestroy(s_left);
    cudaStreamDefault(s_right);
}