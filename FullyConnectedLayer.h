#include "Tensor.h"

class FullyConnectedLayer
{
    public:
        Tensor weights;
        float* biases{nullptr};
        std::string activation{""};

        // Default constructor
        FullyConnectedLayer() = default;

        // Parametrized constructor;
        FullyConnectedLayer(size_t number_of_neurons, size_t previous_layer_dimension ,std::string some_activation);

        // Destructor
        ~FullyConnectedLayer()
        {
            delete[] biases;
        }

        // Feedforward
        float* feedForward(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size);
        float* activate(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size);

        // Backpropagation
        float* backpropagation(float* delta_next, float* delta_this, const float& learning_rate, Tensor& w_next, float* z_this, float* a_prev);
};