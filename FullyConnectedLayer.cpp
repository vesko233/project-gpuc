#include"FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(size_t number_of_neurons, size_t previous_layer_dimension , const std::string& some_activation, const std::string& filename)
{
    if (number_of_neurons < 1 || previous_layer_dimension < 1){
        std::cerr << "Number of neurons cannot be inferior to 1!";
        throw("Invalid input!");
    }
    if (some_activation != "None" && some_activation != "ReLu"){
        std::cerr << "Could not find activation function!";
        throw("Invalid activation function");
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

            delim = "\n";
            while ((pos = row_string.find(delim)) != std::string::npos) {
                row_string.replace(pos,3," ");
            }

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

    // std::cout << weights << std::endl;
    // for (int i = 0; i < number_of_neurons; i++){
    //     std::cout << biases[i] << "; ";
    // }
    // std::cout << std::endl;

    
    // Activation function
    activation = some_activation;
}


// Feed forward data through layer
// Input data is a^(l-1), output data is z^l
// Input data should have size = weights.cols !
// Output data should have size = weights.rows !
void FullyConnectedLayer::feedForward(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size, bool useGPU)
{
    if (input_data_size != weights.get_cols()){
        std::cerr << "Input size of the layer must be equal to the initialized size!";
        throw("Invalid input size");
    }
    if (output_data_size != weights.get_rows()){
        std::cerr << "Output size of the layer must be equal to the initialized size!";
        throw("Invalid output size");
    }

    if (useGPU)
    {
        runFeedForwardGPU(input_data, output_data);
    }
    else runFeedForwardCPU(input_data, output_data);
}

void FullyConnectedLayer::runFeedForwardCPU(float* input_data, float* output_data)
{
    // Multiplying input by weights and adding biases
    for (int i = 0; i < weights.get_rows(); i++){
        float temp = 0;
        for (int j = 0; j < weights.get_cols(); j++){
            temp += weights(i,j)*input_data[j];
        }
        output_data[i] = temp + biases[i];
    }
}

void FullyConnectedLayer::runFeedForwardGPU(float* input_data, float* output_data)
{
    unsigned int size = weights.get_rows()*weights.get_cols()*weights.get_layers();
    float* flatten_weights = new float [size];
    weights.flatten(flatten_weights, size);
    unsigned int N = weights.get_cols();
    unsigned int M = weights.get_rows();
    std::cout << "M = " << M << " N = " << N << std::endl;
    matvec_kernel_cuda(input_data, flatten_weights, biases, output_data, N, M);
    delete [] flatten_weights;
}

// Activation function on layer output
// input data is z^l, output data is a^l
void FullyConnectedLayer::activate(float* input_data, float* output_data, size_t input_data_size, size_t output_data_size)
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