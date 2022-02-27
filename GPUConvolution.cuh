#include "Tensor.h"
#include "ConvolutionLayer.h"
#include "CudaConvolution_16.cuh"
#include "CudaConvolution_32.cuh"

// Function which performs forward pass on some convolutional layer using parallel code
Tensor GPUconvolutuionFeedForward(ConvolutionLayer conv_layer, Tensor& input);