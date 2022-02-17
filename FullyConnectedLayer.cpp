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

    // Initialize random weights from a Gaussian distribution with mean = 0 and stdev = 0.1
    std::default_random_engine generator;
    std::normal_distribution<float> norm_distribution(0.0,0.1);

    Tensor some_weights(number_of_neurons, previous_layer_dimension, 1);
    weights = some_weights;
    biases = new float [number_of_neurons];
    for (int i = 0; i < weights.get_rows(); i++){
        biases[i] = norm_distribution(generator);
        for (int j = 0; j < weights.get_cols(); j++){
            weights(i,j) = norm_distribution(generator);
        }
    }

    // Activation function
    activation = some_activation;
}


// Feed forward data through layer
// Input data is a^(l-1), output data is z^l
// Input data should have size = weights.cols !
// Output data should have size = weights.rows !
float* FullyConnectedLayer::feedForward(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size)
{
    if (input_data_size != weights.get_cols()){
        std::cerr << "Input size of the layer must be equal to the initialized size!";
        throw("Invalid input size");
    }
    if (output_data_size != weights.get_rows()){
        std::cerr << "Output size of the layer must be equal to the initialized size!";
        throw("Invalid output size");
    }

    // Multiplying input by weights and adding biases
    for (int i = 0; i < weights.get_rows(); i++){
        float temp = 0;
        for (int j = 0; j < weights.get_cols(); j++){
            temp += weights(i,j)*input_data[j];
        }
        output_data[i] = temp + biases[i];
    }
    return output_data;
}

// Activation function on layer output
// input data is z^l, output data is a^l
float* FullyConnectedLayer::activate(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size)
{
    if (input_data_size != output_data_size){
        std::cerr << "Activation function dimensions do not match!";
        throw("Invalid dimensions!");
    }

    // Activating input accorfing to specified activation function
    for (int i = 0; i < output_data_size; i++){
        if (activation == "None"){
            output_data[i] = input_data[i];
        } else if (activation == "ReLu"){
            output_data[i] = std::max(input_data[i],0.0f);
        }        
    }
    return output_data;
}

// Backpropagation of layer 
// This method computes this layer's error to be propagated backwards
// Delta_next is the error of the next layer and should have size = number of neurons in next layer ( = delta^l)
// Delta_this is the error of this layer and should have size = number of neurons in this layer ( = delta^(l+1))
// a_prev is the activation of the previous layer and should have size = weights.cols ( = a^(l-1))
// z_this is the non activated output of this layer and should have size = number of neurons in this layer ( = z^l) 
// w_next are the weights of the next layer and shoulf have size = number of neurons in next layer X number of neurons in this layer ( = w^(l+1))
float* FullyConnectedLayer::backpropagation(float* delta_next, float* delta_this, const float& learning_rate, Tensor& w_next, float* z_this, float* a_prev)
{
    // Computing error of this layer
    for (int i = 0; i < weights.get_rows(); i++){
        // Performing scalar product between columns of net layer weights and next layer error
        float temp = 0;
        for (int j = 0; j < w_next.get_rows(); j++){
            temp += w_next(j,i)*delta_next[j];
        }

        // Multiplying by derivative of activation function
        if (activation == "None"){
            delta_this[i] = temp;

        // ReLu derivative is the heaviside function
        } else if (activation == "ReLu"){
            (z_this[i] < 0.0f) ? delta_this[i] = 0.0f : delta_this[i] = temp;
        }
    }

    // Apply gradient descent update rule for weights and biases based on the computed error
    for (int i = 0; i < weights.get_rows(); i ++){
        biases[i] -= learning_rate*delta_this[i];
        for (int j = 0; j < weights.get_cols(); j++){
            weights(i,j) -= learning_rate*delta_this[i]*a_prev[j];
        }
    }

    return delta_this;
}