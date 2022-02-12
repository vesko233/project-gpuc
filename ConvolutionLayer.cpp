#include "ConvolutionLayer.h"

// Parametrized constructor for the convolution layer
ConvolutionLayer::ConvolutionLayer(size_t some_kernel_size, size_t some_kernel_depth, size_t some_stride, size_t some_number_of_kernels)
{
    if (some_kernel_size < 0 || some_kernel_depth < 0 || some_number_of_kernels < 0 || some_stride < 1 ) {
		std::cerr << "The kernel size and/or the number of kernels must be positive; the stride mustn't be smaller than 1!" ;
		throw("Negative values!");
    }

    kernel_size = some_kernel_size;
    kernel_depth = some_kernel_depth;
    stride = some_stride;
    number_of_kernels = some_number_of_kernels;

    // Initialize kernels with values drawn from a normal distribution with mean = 0 and stdev = 1
    std::default_random_engine generator;
    std::normal_distribution<float> norm_distribution(0.0,1.0);

    for (int k = 0; k < number_of_kernels; k++){
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
Tensor ConvolutionLayer::convolution(Tensor& image, Tensor& kernel, const size_t& output_size)
{
    Tensor output(output_size, output_size, 1);

    // Convolution
    // iterate over all output pixels of the image
    int image_size = image.get_rows();
    int offset = (kernel_size - 1)/2;
    for (int i = offset; i < image_size - offset; i++){
        for (int j = offset; j < image_size - offset; j++){

            // Compute convolution result for each pixel
            int res = 0;
            // iterate over layers
            for (int k = 0; k < kernel.get_layers(); k++){
                // convolution
                for (int r = 0; r < kernel_size; r++){
                    for (int c = 0; c < kernel_size; c++){
                        res += kernel(r,c,k)*image(i - (kernel_size-1)/2 + r ,j - (kernel_size-1)/2 + c,k);        
                    }
                }
            }

            // Assign convolution result
            output(i - offset, j - offset, 0) = res;
        }
    }
    return output;
}

// Feed forward function of convolution layer taking an input of size 
Tensor ConvolutionLayer::feedForward(Tensor& input)
{
    // Check if the image depth is the same as the kernel depth
    if (input.get_layers() != kernel_depth){
        std::cerr << "Input depth is not equal to the kernel depth!";
        throw("Invalid dimensions!");
    }

    size_t output_size = (input.get_rows() - kernel_size)/stride + 1;
    Tensor layer_output(output_size, output_size, number_of_kernels);

    // Iterating through all kernels 
    for (int nk = 0; nk < number_of_kernels; nk++){

        // 2D output of a single convolution; parameters[nk] is each kernel in the vector of kernels
        Tensor single_output = convolution(input, parameters[nk], output_size);

        // Copy output of the convolution to the layer output
        for (int i = 0; i < output_size; i++){
            for (int j = 0; j < output_size; j++){
                layer_output(i,j,nk) = single_output(i,j,0);
            }
        }
    }
    return layer_output;
}

