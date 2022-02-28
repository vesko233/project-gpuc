#include "Tensor.h"
#include "CudaGEMV.cuh"

class FullyConnectedLayer
{
    public:
        Tensor weights;
        float* flatten_weights;
        float* biases{nullptr};
        std::string activation;

        // Default constructor
        FullyConnectedLayer() = default;

        // Parametrized constructor;
        FullyConnectedLayer(size_t number_of_neurons, size_t previous_layer_dimension , const std::string& some_activation, const std::string& filename);

        // Destructor
        ~FullyConnectedLayer()
        {
            delete biases;
        }

        // Feedforward
        void feedForward(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size, bool useGPU);
        void activate(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size);
        void runFeedForwardCPU(float* input_data, float* output_data);
        void runFeedForwardGPU(float* input_data, float* output_data);
        // Backpropagation
        float* backpropagation(float* delta_next, float* delta_this, const float& learning_rate, Tensor& w_next, float* z_this, float* a_prev);
};