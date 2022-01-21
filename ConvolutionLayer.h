#include "Tensor.h"

class ConvolutionLayer
{
    private:
        // Parameter 
        std::vector<Tensor> parameters;
        size_t kernel_depth{0};
        size_t kernel_size{0};
        size_t stride{0};
        size_t number_of_kernels{0};

    public:
        // Default constructor
        ConvolutionLayer() = default;

        // Parametrized constructor
        ConvolutionLayer(size_t some_kernel_size, size_t some_kernel_depth, size_t some_stride, size_t some_number_of_kernels);

        // Destructor
        ~ConvolutionLayer()
        {}

        // Convolution between an image and a kernel
        Tensor convolution(Tensor& image, Tensor& kernel);

        // Feed forward
        Tensor feedForward(Tensor& input);
};