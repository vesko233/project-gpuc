#pragma once
#include "Tensor.h"

class ConvolutionLayer
{
    public:
        // Parameter 
        std::vector<Tensor> parameters;
        float* biases;
        size_t kernel_depth{0};
        size_t kernel_size{0};
        size_t stride{1};
        size_t number_of_kernels{0};
        std::string activation{""};

        // Default constructor
        ConvolutionLayer() = default;

        // Parametrized constructor
        ConvolutionLayer(size_t some_kernel_size, size_t some_kernel_depth, size_t some_stride, size_t some_number_of_kernels, const std::string& some_activation, const std::string& filename);

        // Destructor
        ~ConvolutionLayer()
        {
            delete[] biases;
        }

        // Convolution between an image and a kernel
        Tensor convolution(Tensor& image, Tensor& kernel, const size_t& output_size);

        // Feed forward
        Tensor feedForward(Tensor& input);

        // Obtain flat array of all kernels
        void flatten_kernels(float* flat_kernels, size_t flat_kernels_size);
};