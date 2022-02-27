#include "CudaConvolution_32.cuh"

// GRID DIMENSIONS = 16 x 32
// BLOCK DIMENSIONS = 30x30 = 900
// blockIdx.y in the kernel below refers to the layer of the input tensor and filter, i.e. the z dimension
__global__ void convolution2D_kernel_32(float* d_image, float* d_filter, float* d_output, size_t output_size)
{
    // Allocating shared memory in each block for the entire input image referring to this blockIdx
    // Each block of threads will be performing a 2D convolution of a layer of the input image with the corresponding layer of the filter
    // The grid will have dimensions 16 x 32, referring to the fact that we will perform convolutions with 32 3x3x16 filters.
    __shared__ float shared_image[BLOCK_SIZE_2][BLOCK_SIZE_2];
    __shared__ float shared_filter[FILTER_SIZE_2][FILTER_SIZE_2];

    // Fetching from global memory layer of image, depending on the blockIdx.y
    shared_image[threadIdx.x][threadIdx.y] = d_image[threadIdx.x + threadIdx.y*BLOCK_SIZE_2 + blockIdx.y*BLOCK_SIZE_2*BLOCK_SIZE_2];

    // Fetching from global memory layer of filter, depending on the blockIdx.y
    if (threadIdx.x < FILTER_SIZE_2 && threadIdx.y < FILTER_SIZE_2){
        shared_filter[threadIdx.x][threadIdx.y] = d_filter[threadIdx.x + threadIdx.y*FILTER_SIZE_2 + blockIdx.y*FILTER_SIZE_2*FILTER_SIZE_2 + blockIdx.x*FILTER_SIZE_2*FILTER_SIZE_2*DEPTH_2];
    }

    __syncthreads();

    float Res = 0;
    if (threadIdx.y < output_size && threadIdx.x < output_size){
        // Each thread performing convolution with filter
        for (int i = 0; i < FILTER_SIZE_2; i++){
            for (int j = 0; j < FILTER_SIZE_2; j++){
                Res += shared_image[threadIdx.y + i][threadIdx.x + j]*shared_filter[i][j];
            }
        }
        // Adding result to output. Here, we are using atomic add to avoid errors
        atomicAdd(&(d_output[threadIdx.y + output_size*threadIdx.x + blockIdx.x*output_size*output_size]),Res);
    }

}


// image = Flattened 3D tensor (30x30x16)
// filter = Flattened 4D tensor (3x3x16x32)
// output = Flattened 3D tensor (28x28x32)
void convolution_2D_32(float* image, float* filter, float* output)
{
    // 32 x 32 block
    dim3 dimBlock(BLOCK_SIZE_2,BLOCK_SIZE_2,1);

    // 3 x 16 grid
    dim3 dimGrid(FILTERS_NUM_2,DEPTH_2,1);

    // Declaring device copies of data
    float* d_image;
    float* d_filter;
    float* d_output;

    // Initializing memory sizes for image, filter and output
    int image_memory_size = BLOCK_SIZE_2*BLOCK_SIZE_2*DEPTH_2*sizeof(float); // 30 x 30 x 16 x (float)
    int filter_memory_size = FILTER_SIZE_2*FILTER_SIZE_2*DEPTH_2*FILTERS_NUM_2*sizeof(float); // 3 x 3 x 16 x 32 x (float)
    int output_memory_size = (BLOCK_SIZE_2 - FILTER_SIZE_2 + 1)*(BLOCK_SIZE_2 - FILTER_SIZE_2 + 1)*FILTERS_NUM_2*sizeof(float); // 28 x 28 x 32 x (float)

    // Allocating memory on device
    cudaMalloc((void**) &d_image, image_memory_size);
    cudaMalloc((void**) &d_filter, filter_memory_size);
    cudaMalloc((void**) &d_output, output_memory_size);

    // Copying memory on device from host
    cudaMemcpy(d_image, image, image_memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filter_memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, output_memory_size, cudaMemcpyHostToDevice);
    
    // Executng the kernel
    convolution2D_kernel_32<<<dimGrid,dimBlock>>>(d_image,d_filter,d_output,BLOCK_SIZE_2-FILTER_SIZE_2+1);

    // Copy result back to host memory from device memory
    cudaMemcpy(output, d_output, output_memory_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_output);
}


/////////////////////////////////////////////////////////////////////////
