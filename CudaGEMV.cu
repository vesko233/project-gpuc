#include "CudaGEMV.cuh"
#define THREADS_PER_BLOCK 64 // For kernelRD
//#define THREADS_PER_BLOCK 1024 // For kernelST
#define blockSize 64         // For kernelRD

__device__ void warpReduce(volatile float *sdata, unsigned int tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8 ];
    if (blockSize >= 8 ) sdata[tid] += sdata[tid + 4 ];
    if (blockSize >= 4 ) sdata[tid] += sdata[tid + 2 ];
    if (blockSize >= 2 ) sdata[tid] += sdata[tid + 1 ];

}

// Reduction kernel
__global__ void kernelRD(float *vec, float *mat, float *b, float *out, const unsigned int N, const unsigned int M)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    unsigned int tidd = threadIdx.x+blockIdx.x*blockDim.x;
    sdata[tid] = 0;
    if(tidd < N)
    {
        while (i < N) {
            sdata[tid] += vec[i]*mat[i + blockIdx.y*N] + vec[i + blockSize]*mat[i + blockSize + blockIdx.y*N];
            i += gridSize;
        }
        __syncthreads();
        if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
        if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
        if (blockSize >= 128) { if (tid < 64 ) { sdata[tid] += sdata[tid + 64 ]; } __syncthreads(); }
        if (tid < 32) warpReduce(sdata, tid);
        __syncthreads();
        if (tid == 0)
            atomicAdd(&out[blockIdx.y], sdata[0]);
        __syncthreads();
        if (tid == 0 && blockIdx.x == 0)
            atomicAdd(&out[blockIdx.y], b[blockIdx.y]);
    }
}

// Naive kernel
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

    cudaMalloc((void**)&dev_input , sizeof(float)*N  );
    cudaMalloc((void**)&dev_matrix, sizeof(float)*N*M);
    cudaMalloc((void**)&dev_biases, sizeof(float)*M  );
    cudaMalloc((void**)&dev_output, sizeof(float)*M  );

    for(int i = 0; i < M; ++i) output[i] = 0.0f;

    cudaMemcpy(dev_input,  input,  sizeof(float)*N,   cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix, matrix, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_biases, biases, sizeof(float)*M,   cudaMemcpyHostToDevice);
    cudaMemcpy(dev_output, output, sizeof(float)*M,   cudaMemcpyHostToDevice);

    // Reduction kernel
    // 1024 x 1 blocks
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    // In M x N/1024+1 grid
    dim3 dimGrid(N/THREADS_PER_BLOCK + 1, M, 1);
    kernelRD<<<dimGrid, dimBlock, THREADS_PER_BLOCK*sizeof(float)>>>(dev_input, dev_matrix, dev_biases, dev_output, N, M);

    // Naive kernel
    // kernelST<<<1,M>>>(dev_input, dev_matrix, dev_biases, dev_output, N, M);

    cudaMemcpy(output, dev_output, sizeof(float)*M, cudaMemcpyDeviceToHost);

    cudaFree(dev_input );
    cudaFree(dev_matrix);
    cudaFree(dev_biases);
    cudaFree(dev_output);
}




