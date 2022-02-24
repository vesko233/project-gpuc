#ifndef GPU_PROJECT_GPUTILITIES_CUH
#define GPU_PROJECT_GPUTILITIES_CUH
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 32
__global__ void kernel(float *vec, float *mat, float *out, const unsigned int N, const unsigned int M);

void matvec_kernel_cuda(float* a, float* b, float* c, unsigned int N, unsigned int M);

#endif //GPU_PROJECT_GPUTILITIES_CUH
