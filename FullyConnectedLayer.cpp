#include"FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(size_t number_of_neurons, size_t previous_layer_dimension ,std::string some_activation)
{
    if (number_of_neurons < 1 || previous_layer_dimension < 1){
        std::cerr << "Number of neurons cannot be inferior to 1!";
        throw("Invalid input!");
    }
    if (some_activation != "None" && some_activation != "ReLu"){
        std::cerr << "Could not find activation function!";
        throw("Invalid activation function");
    }

    // Initialize random weights from a Gaussian distribution with mean = 0 and stdev = 1
    std::default_random_engine generator;
    std::normal_distribution<float> norm_distribution(0.0,1.0);

    Tensor some_weights(number_of_neurons, previous_layer_dimension, 1);
    weights = some_weights;
    biases = new float [number_of_neurons];
    for (int i = 0; i < weights.get_rows(); i++){
        biases[i] = norm_distribution(generator);
        for (int j = 0; j < weights.get_cols(); j++){
            weights(i,j,0) = norm_distribution(generator);
        }
    }

    // Activation function
    activation = some_activation;
}

// Feed forward data through layer
float* FullyConnectedLayer::feedForward(float* input_data, size_t input_data_size)
{
    if (input_data_size != weights.get_cols()){
        std::cerr << "Input size of the layer must be equal to the initialized size!";
        throw("Invalid input size");
    }

    // Multiplying input by wrights and adding bias
    float layer_output[weights.get_rows()];
    for (int i = 0; i < weights.get_rows(); i ++){
        float res = 0;
        for (int j = 0; j < input_data_size; j++){
            res += weights(i,j,0)*input_data[j];
        }
        res += biases[i];
        layer_output[i] = res;
    }

    return layer_output;
}

// Activation function on layer output
float* FullyConnectedLayer::activate(float* z)
{
    float activate_output[weights.get_rows()];
    for (int i = 0; i < weights.get_rows(); i++){
        if (activation == "None"){
            activate_output[i] = z[i];
        } else if (activation == "ReLu"){
            activate_output[i] = std::max(z[i],0.0f);
        }
    }
    return activate_output;
}

// Backpropagation of layer
float* FullyConnectedLayer::backpropagation(float* input_data, float learning_rate, float* next_layer_error, float* z)
{
    float output_error[weights.get_cols()];

    // Computing error of this layer
    // iterate through number of neurons in previous layer (weights transpose)
    for (int i = 0; i < weights.get_cols(); i++){
        float res = 0;
        for (int j = 0; j < weights.get_rows(); j++){
            res += weights(j,i,0)*next_layer_error[j];
        }

        if (activation == "None"){
            output_error[i] = res;
        } else if (activation == "ReLu"){
            (z[i] < 0.0f) ? output_error[i] = 0.0f : output_error[i] = res;
        }
    }

    // Apply gradient descent to weights and biases
    for (int i = 0; i < weights.get_rows(); i ++){
        biases[i] -= learning_rate*output_error[i];
        for (int j = 0; j < weights.get_cols(); j++){
            weights(i,j,0) -= learning_rate*output_error[i]*input_data[j];
        }
    }

    return output_error;
}