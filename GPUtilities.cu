#include "GPUtilities.cuh"
__global__
void kernel(float *vec, float *mat, float *out, const unsigned int N, const unsigned int M)
{
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    float sum=0;
    if(tid<M){
        for(int i=0; i<N; i++)
            sum += vec[i]*mat[(i*M)+tid];
        out[tid]=sum;
    }
}

void matvec_kernel_cuda(float* a, float* b, float* c, unsigned int N, unsigned int M)
{
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, sizeof(float)*N);
    cudaMalloc((void**)&dev_b, sizeof(float)*N*M);
    cudaMalloc((void**)&dev_c, sizeof(float)*M);

    cudaMemcpy(dev_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float)*N*M, cudaMemcpyHostToDevice);

    printf("\n\nRunning Kernel...\n\n");
    kernel<<<M/256+1, 256>>>(dev_a, dev_b, dev_c, N, M);
    //printf("error code: %s\n",cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(c, dev_c, sizeof(float)*M, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}


