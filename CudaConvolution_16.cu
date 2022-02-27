#include "CudaConvolution_16.cuh"

// GRID DIMENSIONS = 3 x 16
// BLOCK DIMENSIONS = 32x32 = 1024
// blockIdx.y in the kernel below refers to the layer of the input tensor and filter, i.e. the z dimension
__global__ void convolution2D_kernel_16(float* d_image, float* d_filter, float* d_output, size_t output_size)
{
    // Allocating shared memory in each block for the entire input image referring to this blockIdx
    // Each block of threads will be performing a 2D convolution of a layer of the input image with the corresponding layer of the filter
    // The grid will have dimensions 3 x 16, referring to the fact that we will perform convolutions with 16 3x3x3 filters.
    __shared__ float shared_image[BLOCK_SIZE_1][BLOCK_SIZE_1];
    __shared__ float shared_filter[FILTER_SIZE_1][FILTER_SIZE_1];

    // Fetching from global memory layer of image, depending on the blockIdx.y
    shared_image[threadIdx.x][threadIdx.y] = d_image[threadIdx.x + threadIdx.y*BLOCK_SIZE_1 + blockIdx.y*BLOCK_SIZE_1*BLOCK_SIZE_1];

    // Fetching from global memory layer of filter, depending on the blockIdx.y
    if (threadIdx.x < FILTER_SIZE_1 && threadIdx.y < FILTER_SIZE_1){
        shared_filter[threadIdx.x][threadIdx.y] = d_filter[threadIdx.x + threadIdx.y*FILTER_SIZE_1 + blockIdx.y*FILTER_SIZE_1*FILTER_SIZE_1 + blockIdx.x*FILTER_SIZE_1*FILTER_SIZE_1*DEPTH_1];
    }

    __syncthreads();

    float Res = 0.0f;
    if (threadIdx.y < output_size && threadIdx.x < output_size){
        // Each thread performing convolution with filter
        for (int i = 0; i < FILTER_SIZE_1; i++){
            for (int j = 0; j < FILTER_SIZE_1; j++){
                Res += shared_image[threadIdx.x + i][threadIdx.y + j]*shared_filter[i][j];
            }
        }
        // Adding result to output. Here, we are using atomic add to avoid errors
        atomicAdd(&(d_output[threadIdx.x + output_size*threadIdx.y + blockIdx.x*output_size*output_size]), Res);
    }

}


// image = Flattened 3D tensor (32x32x3)
// filter = Flattened 4D tensor (3x3x3x16)
// output = Flattened 3D tensor (30x30x16)
void convolution_2D_16(float* image, float* filter, float* output)
{
    // 32 x 32 block
    dim3 dimBlock(BLOCK_SIZE_1,BLOCK_SIZE_1,1);

    // 16 x 3 grid
    dim3 dimGrid(FILTERS_NUM_1,DEPTH_1,1);

    // Declaring device copies of data
    float* d_image;
    float* d_filter;
    float* d_output;

    // Initializing memory sizes for image, filter and output
    int image_memory_size = BLOCK_SIZE_1*BLOCK_SIZE_1*DEPTH_1*sizeof(float); // 32 x 32 x 3 x (float)
    int filter_memory_size = FILTER_SIZE_1*FILTER_SIZE_1*DEPTH_1*FILTERS_NUM_1*sizeof(float); // 3 x 3 x 3 x 16 x (float)
    int output_memory_size = (BLOCK_SIZE_1 - FILTER_SIZE_1 + 1)*(BLOCK_SIZE_1 - FILTER_SIZE_1 + 1)*FILTERS_NUM_1*sizeof(float); // 30 x 30 x 16 x (float)

    for(int i = 0; i < output_memory_size/sizeof(float); ++i) output[i] = 0.0f;

    // Allocating memory on device
    cudaMalloc((void**) &d_image, image_memory_size);
    cudaMalloc((void**) &d_filter, filter_memory_size);
    cudaMalloc((void**) &d_output, output_memory_size);

    // Copying memory on device from host
    cudaMemcpy(d_image, image, image_memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filter_memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, output_memory_size, cudaMemcpyHostToDevice);
    
    // Executng the kernel
    convolution2D_kernel_16<<<dimGrid,dimBlock>>>(d_image,d_filter,d_output,BLOCK_SIZE_1-FILTER_SIZE_1+1);

    // Copy result back to host memory from device memory
    cudaMemcpy(output, d_output, output_memory_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_output);
}


/////////////////////////////////////////////////////////////////////////
