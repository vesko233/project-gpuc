#ifndef CUDA_CONVOLUTION_32_CUH
#define CUDA_CONVOLUTION_32_CUH
#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <time.h>

#define BLOCK_SIZE_2 30
#define FILTER_SIZE_2 3
#define FILTERS_NUM_2 32
#define DEPTH_2 16

// GRID DIMENSIONS = 16 x 32
// BLOCK DIMENSIONS = 30x30 = 900
// blockIdx.y in the kernel below refers to the layer of the input tensor and filter, i.e. the z dimension
__global__ void convolution2D_kernel_32(float* d_image, float* d_filter, float* d_output, size_t output_size); 


// image = Flattened 3D tensor (30x30x16)
// filter = Flattened 4D tensor (3x3x16x32)
// output = Flattened 3D tensor (28x28x32)
void convolution_2D_32(float* image, float* filter, float* output);


#endif