#include "SoftmaxLayer.h"

SoftmaxLayer::SoftmaxLayer(size_t number_of_neurons, size_t previous_layer_dimension)
{
    if (number_of_neurons < 1 || previous_layer_dimension < 1){
        std::cerr << "Number of neurons cannot be inferior to 1!";
        throw("Invalid input!");
    }

    // Initialize random weights from a Gaussian distribution with mean = 0 and stdev = 0.1
    std::default_random_engine generator;
    std::normal_distribution<float> norm_distribution(0.0,0.1);

    weights = Tensor(number_of_neurons, previous_layer_dimension, 1);
    biases = new float [number_of_neurons];
    for (int i = 0; i < weights.get_rows(); i++){
        biases[i] = norm_distribution(generator);
        for (int j = 0; j < weights.get_cols(); j++){
            weights(i,j) = norm_distribution(generator);
        }
    }
}


float* SoftmaxLayer::feedForward(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size)
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


// Softmax activation funciton
// Input data should have size = number of neurons ( = z^L)
// Output data should have size = number of neurons ( = a^L) 
float* SoftmaxLayer::softmaxActivate(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size)
{
    if (input_data_size != output_data_size){
        std::cerr << "Activation function dimensions do not match!";
        throw("Invalid dimensions!");
    }

    // Calculating sum of exponentials of the input data
    float exp_sum = 0;
    for (int i = 0; i < input_data_size; i++){
        exp_sum += std::exp(input_data[i]);
    }

    // Obtaining output 
    for (int i = 0; i < input_data_size; i++){
        output_data[i] = std::exp(input_data[i])/exp_sum;
    }

    return output_data;
}


// Backpropagation of softmax layer. This should be the last layer of the neural network.
// Since it is the last layer, the gradient is computed a bit differently than the gradient of a regular 
// layer in the neural network. It is asssumed that 
float* SoftmaxLayer::backpropagation(float* delta_this, float* labels, const float& learning_rate, float* z_this, float* a_this, float* a_prev)
{
    // Recompute exp_sum with one operation
    float exp_sum = std::exp(z_this[0])/a_this[0];

    // Computing the error of this layer
    for (int i = 0; i < weights.get_rows(); i++){
        float e_z = std::exp(z_this[i]);
        delta_this[i] = ((a_this[i] - labels[i])*(exp_sum - e_z)*e_z)/std::pow(exp_sum,2);
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
