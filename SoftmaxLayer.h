#include "Tensor.h"
#include "CudaGEMV.cuh"

class SoftmaxLayer
{
    public:
        Tensor weights;
        float* flatten_weights;
        float* biases{nullptr};

        // Default constructor
        SoftmaxLayer() = default;

        // Parametrized constructor;
        SoftmaxLayer(size_t number_of_neurons, size_t previous_layer_dimension, const std::string& filename);

        // Destructor
        ~SoftmaxLayer()
        {
            delete[] biases;
        }

        // Feedforward 
        void feedForward(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size, bool useGPU);
        void softmaxActivate(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size);
        void runFeedForwardCPU(float* input_data, float* output_data);
        void runFeedForwardGPU(float* input_data, float* output_data);
        // Backpropagation
        float* backpropagation(float* delta_this, float* labels, const float& learning_rate, float* z_this, float* a_this, float* a_prev);
};