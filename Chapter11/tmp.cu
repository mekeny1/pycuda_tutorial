__device__ void __inline__ laneid(int *id)
{
    asm("mov.u32 %0,%%laneid;":"=r"(*id));
}

__device__ void __inline__ split64(double val,int *lo,int *hi)
{
    asm volatile("mov.b64 {%0 %1},%2;":"=r"(*lo),"=r"(*hi):"d"(*val));
}

__device__ void __inline__ combine64(double *val,int lo,int hi)
{
    asm volatile("mov.b64 %0,{%1,%2};":"=d"(*val):"r"(lo),"r"(hi));
}

__global__ void sum_ker(double *input,double *out)
{
    int id;
    laneid(id);

    double2 vals=*reinterpret_cast<double2*>(&input[(blockDim.x*blockIdx.x+threadIdx.x)*2]);

    double sum_val=vals.x+vals.y;
    double temp;

    int s1,s2;

    for (int i = 1; i < 32; i*2)
    {
        split64(sum_val,s1,s2);

        s1=__shfl_down_sync(s1,i,32);
        s2=__shfl_down_sync(s2,i,32);

        combine64(temp,s1,s2);
        sum_val+=temp;
    }

    if(id==0)
    {
        atomicAdd(out,sum_val);
    }
}
