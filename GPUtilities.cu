#include "GPUtilities.cuh"
#define THREADS_PER_BLOCK 512
#define BLOCKSIZE 32

//__global__
//void kernel(float *vec, float *mat, float *b, float *out, const unsigned int N, const unsigned int M)
//{
//    int tid=threadIdx.x+blockIdx.x*blockDim.x;
//    __shared__ float* smat = new float[M];
//    for(int i=0; i<M; i++) smat[i] = mat[(i*N)+tid];
//
//    __shared__ float *sum = new float[N*M];
//    if(tid<N){
//        for(int i=0; i<M; i++)
//            sum[tid] = vec[tid]*smat[i];
//        __syncthreads();
//        for(int i=0; i<M; i++)
//            out[i] += sum[i*N + tid];
//        __syncthreads();
//        for(int i=0; i<M; i++)
//            out[i] += b[tid];
//    }
//}

//SLOW BUT TESTED
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
    //printf("Starting MatVectMultiplication...\n");
    float *dev_input, *dev_matrix, *dev_biases, *dev_output;

    cudaMalloc((void**)&dev_input, sizeof(float)*N);
    cudaMalloc((void**)&dev_matrix, sizeof(float)*N*M);
    cudaMalloc((void**)&dev_biases, sizeof(float)*M);
    cudaMalloc((void**)&dev_output, sizeof(float)*M);

    cudaMemcpy(dev_input, input, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix, matrix, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_biases, biases, sizeof(float)*M, cudaMemcpyHostToDevice);

    //printf("\n\nRunning Kernel...\n\n");
    int max=BLOCKSIZE*BLOCKSIZE;
    int BlocksPerGrid=N/max+1;
    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
    if(N%max==0)BlocksPerGrid--;
    dim3 dimGrid(1,BlocksPerGrid);
    kernelST<<<1,M>>>(dev_input, dev_matrix, dev_biases, dev_output, N, M);
    //kernel<<<1,M>>>(dev_input, dev_matrix, dev_biases, dev_output, N, M);
    //printf("error code: %s\n",cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(output, dev_output, sizeof(float)*M, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_matrix);
    cudaFree(dev_biases);
    cudaFree(dev_output);
}


