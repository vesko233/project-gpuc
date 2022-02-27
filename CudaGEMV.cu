#include "CudaGEMV.cuh"

__global__
void kernelST(float *vec, float *mat, float *b, float *out, const unsigned int N, const unsigned int M)
{
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    float sum=0.0f;
    if(tid<M){
        for(int i=0; i<N; i++)
            sum += vec[i]*mat[(tid*N)+i];
        out[tid]=sum + b[tid];
    }
    __syncthreads();
}

void matvec_kernel_cuda(float* input, float* matrix, float* biases, float* output,  unsigned int N, unsigned int M)
{
    float *dev_input, *dev_matrix, *dev_biases, *dev_output;

    cudaMalloc((void**)&dev_input, sizeof(float)*N);
    cudaMalloc((void**)&dev_matrix, sizeof(float)*N*M);
    cudaMalloc((void**)&dev_biases, sizeof(float)*M);
    cudaMalloc((void**)&dev_output, sizeof(float)*M);

    cudaMemcpy(dev_input, input, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix, matrix, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_biases, biases, sizeof(float)*M, cudaMemcpyHostToDevice);

    kernelST<<<1,M>>>(dev_input, dev_matrix, dev_biases, dev_output, N, M);

    cudaMemcpy(output, dev_output, sizeof(float)*M, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_matrix);
    cudaFree(dev_biases);
    cudaFree(dev_output);
}




