#include <curand_kernel.h>

#define ULL unsigned long long
#define _R(z) (1.0f / (z))
#define _P2(z) ((z) * (z))

__device__ inline % (p)s f(% (p)s x)
{
    % (p)s y;
    % (matach_function)s;

    return y;
}

extern "C"
{
    __global__ void monte_carlo(int iters, % (p)s lo, % (p)s hi, % (p)s * ys_out)
    {
        curandState cr_state;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int num_threads = blockDim.x * gridDim.x;

        % (p)s t_width = (hi - lo) / (% (p)s) num_threads;

        % (p)s density = ((% (p)s) iters) / t_width;

        % (p)s t_lo = t_width * tid + lo;
        % (p)s t_hi = t_lo + t_width;

        curand_init((ULL)clock() + (ULL)tid, (ULL)0, (ULL)0, &cr_state);

        % (p)s y, y_sum = 0.0f;
        % (p)s rand_val, x;

        for (int i = 0; i < iters; i++)
        {
            rand_val = curand_uniform % (p_curand)s(&cr_state);
            x = t_lo + t_width * rand_val;
            y_sum += f(x);
        }

        y = y_sum / density;
    }
}
