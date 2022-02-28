#ifndef GPU_PROJECT_CUDAGEMV_CUH
#define GPU_PROJECT_CUDAGEMV_CUH

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void kernelST(float *vec, float *mat, float* biases, float *out, const unsigned int N, const unsigned int M);
__global__ void kernelRD(float *vec, float *mat, float* biases, float *out, const unsigned int N, const unsigned int M);

void matvec_kernel_cuda(float* input, float* matrix, float* biases, float* output,  unsigned int N, unsigned int M);

#endif //GPU_PROJECT_CUDAGEMV_CUH
