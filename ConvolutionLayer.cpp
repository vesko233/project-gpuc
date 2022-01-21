#include "ConvolutionLayer.h"

// Parametrized constructor for the convolution layer
ConvolutionLayer::ConvolutionLayer(size_t some_kernel_size, size_t some_kernel_depth, size_t some_stride, size_t some_number_of_kernels)
{
    if (some_kernel_size < 0 || some_kernel_depth < 0 || some_number_of_kernels < 0 || some_stride < 0 ) {
		std::cerr << "The kernel size, the number of kernels and/or the stride must be positive!" ;
		throw("Negative values!");
    }

    kernel_size = some_kernel_size;
    kernel_depth = some_kernel_depth;
    stride = some_stride;
    number_of_kernels = some_number_of_kernels;

    // Initialize kernels with values drawn from a normal distribution with mean = 0 and stdev = 1
    std::default_random_engine generator;
    std::normal_distribution<float> norm_distribution(0.0,1.0);

    for (int k = 0; k < some_number_of_kernels; k++){
        // Initialize kernel
        Tensor kernel(kernel_size, kernel_size, kernel_depth);
        for (int i = 0; i < kernel_size; i++){
            for (int j = 0; j < kernel_size; j++){
                for (int k = 0; k < some_kernel_depth; k++){
                    kernel(i,j,k) = norm_distribution(generator);
                }
            }
        }
        parameters.push_back(kernel);
    }
}

// Method for convolving an image with a kernel
Tensor ConvolutionLayer::convolution(Tensor& image, Tensor& kernel)
{
    size_t result_size = (image.get_rows() - kernel.get_rows())/stride + 1;
    Tensor result(result_size, result_size, 1);

    // Convolution
    int offset = (kernel_size - 1)/2;
    for (int i = 0; i < result_size; i++){
        for (int j = 0; j < result_size; j++){
            float res = 0;
            for (int k = 1; k <= kernel_size; k + stride){
                for (int l = 1; l <= kernel_size; l + stride){
                    res += image(i - offset + k ,j - offset + l)*kernel(k,l);
                }
            }
            result(i,j) = res;
        }
    }
    return result;
}

// Feed forward function of convolution layer taking an input of size width X height X depth
std::vector<Matrix> ConvolutionLayer::feedForward(std::vector<Matrix>& input)
{
    // For each input image,   
    if (input.size() != kernel_depth){
        std::cerr << "Input size of vector is not equal to the kernel depth!";
        throw("Invalid dimensions!");
    }

    // Define kernel output size
    size_t output_size = (input[0].get_rows() - kernel_size)/stride + 1;

    // For each kernel, compute its convolution with the input vector
    std::vector<Matrix> output;

    for (int i = 0; i < number_of_kernels; i++){
        std::vector<Matrix> kernel = parameters[i];

        Matrix kernel_output(output_size, output_size);
        
        // 2D convolve
        for (int j = 0; j < kernel_depth; j++){
            kernel_output = kernel_output + convolution(input[j], kernel[j]);
        }

        // Add kernel output to output vector
        output.push_back(kernel_output);
    }
    return output;
}

