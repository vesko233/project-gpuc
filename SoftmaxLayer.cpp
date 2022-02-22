#include "SoftmaxLayer.h"

SoftmaxLayer::SoftmaxLayer(size_t number_of_neurons, size_t previous_layer_dimension, const std::string& filename)
{
    if (number_of_neurons < 1 || previous_layer_dimension < 1){
        std::cerr << "Number of neurons cannot be inferior to 1!";
        throw("Invalid input!");
    }

    // Extracting weights and biases from file
    std::ifstream weight_file(filename.c_str());
    if (weight_file.is_open()){
        // Getting entire string of file
        std::string entire_string;
        std::getline(weight_file,entire_string,'=');

        //
        // Splitting string into two sections, one for weights and another for biases
        std::string delimiter = "%"; size_t delim_pos = entire_string.find(delimiter);

        // Extract weight string and erase it
        std::string weight_string = entire_string.substr(0,delim_pos);
        entire_string.erase(0,delim_pos+delimiter.length());

        // Find new position of delimiter and extract bias string
        delim_pos = entire_string.find(delimiter);
        std::string bias_string = entire_string.substr(0,delim_pos);


        //
        // Container variables
        std::string delim; size_t pos; std::string elem;
        std::string row_string;


        //
        // Extracting weights
        weights = Tensor(number_of_neurons, previous_layer_dimension, 1);
        weight_string = weight_string.substr(weight_string.find("[[") + 1,weight_string.find("]]"));

        // For each neuron in the previous layer
        for (int i = 0; i < previous_layer_dimension; i++){
            // Extracting row string
            delim = "]";
            pos = weight_string.find(delim);
            row_string = weight_string.substr(0,pos + delim.length());
            weight_string.erase(0,pos + delim.length());

            // Trimming row string
            pos = row_string.find("[");
            row_string.replace(row_string.find("]"), 1, " ");
            row_string = row_string.substr(pos + 1, row_string.length() - pos);

            delim = " ";
            // For each neuron in this layer
            for (int j = 0; j < number_of_neurons; j++){
                pos = row_string.find(delim);
                elem = row_string.substr(0,pos);
                std::string::size_type sz;
                float elem_f = std::stof(elem, &sz);
                row_string.erase(0, pos + delim.length());
                weights(j,i) = elem_f;
            }
        }


        //
        // Extracting biases
        size_t begin_pos = bias_string.find('['); 
        size_t end_pos = bias_string.find(']');
        bias_string.replace(end_pos, 1, " ");
        bias_string = bias_string.substr(begin_pos + 1, end_pos - begin_pos);

        biases = new float[number_of_neurons];
        delim = " ";
        for (int i = 0; i < number_of_neurons; i++){
            pos = bias_string.find(delim);
            elem = bias_string.substr(0,pos);
            std::string::size_type sz;
            float elem_f = std::stof(elem, &sz);
            bias_string.erase(0, pos + delim.length());
            biases[i] = elem_f;
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
