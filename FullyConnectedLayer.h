#include "Tensor.h"

class FullyConnectedLayer
{
    private:
        Tensor weights;
        float* biases{nullptr};
        std::string activation{""};

    public:
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
        float* feedForward(float* input_data, size_t input_data_size);
        float* activate(float* z);

        // Backpropagation
        float* backpropagation(float* input_data, float learning_rate, float* next_layer_error, float* z);
};