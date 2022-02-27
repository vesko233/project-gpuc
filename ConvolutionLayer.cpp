#include "ConvolutionLayer.h"

// Parametrized constructor for the convolution layer
ConvolutionLayer::ConvolutionLayer(size_t some_kernel_size, size_t some_kernel_depth, size_t some_stride, size_t some_number_of_kernels, const std::string& some_activation, const std::string& filename)
{
    if (some_kernel_size < 0 || some_kernel_depth < 0 || some_number_of_kernels < 0 || some_stride < 1 ) {
		std::cerr << "The kernel size and/or the number of kernels must be positive; the stride mustn't be smaller than 1!" ;
		throw("Negative values!");
    }
    
    if (some_activation != "None" && some_activation != "ReLu"){
        std::cerr << "Activation function does not exist!";
        throw("Invalid input!");
    }

    activation = some_activation;
    kernel_size = some_kernel_size;
    kernel_depth = some_kernel_depth;
    stride = some_stride;
    number_of_kernels = some_number_of_kernels;

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

        // 
        // Extracting weights
        // Initialize vector of Tensors
        for (int i = 0; i < number_of_kernels; i++){
            Tensor temp_tensor(kernel_size,kernel_size,kernel_depth);
            parameters.push_back(temp_tensor);
        }
        weight_string.erase(0,1);

        // declaring container variables
        std::string i_string; std::string j_string; std::string k_string;
        size_t begin_pos; size_t end_pos;
        size_t pos = 0; std::string delim = " "; std::string elem;

        // rows
        for (int i = 0; i < kernel_size; i++){
            // extracting row vals and erasing part of weight_string
            delim = "]]]";
            pos = weight_string.find(delim);
            i_string = weight_string.substr(0,pos + delim.length());
            weight_string.erase(0, pos + delim.length());

            // extracting row string
            begin_pos = i_string.find("[[[");
            i_string = i_string.substr(begin_pos + 1, i_string.find("]]]") - begin_pos + 1);

            // cols
            for (int j = 0; j < kernel_size; j++){
                // extracting col vals and erasing part of i_string
                delim = "]]";
                pos = i_string.find(delim);
                j_string = i_string.substr(0,pos + delim.length());
                i_string.erase(0,pos + delim.length());

                // extracting col string
                begin_pos = j_string.find("[["); 
                j_string = j_string.substr(begin_pos + 1, j_string.find("]]") - begin_pos);

                // layers
                for (int k = 0; k < kernel_depth; k++){
                    // extracting layer vals and erasing part of j_string
                    delim = "]";
                    pos = j_string.find(delim);
                    k_string = j_string.substr(0,pos + delim.length());
                    j_string.erase(0,pos + delim.length());

                    // extracting layer string
                    begin_pos = k_string.find("[");
                    end_pos = k_string.find("]");
                    k_string.replace(end_pos, 1, " ");
                    k_string = k_string.substr(begin_pos + 1, end_pos - begin_pos);

                    // Get rid of new lines
                    pos = k_string.find("\n");
                    while ( pos != std::string::npos){
                        k_string.replace(pos,5," ");
                        pos = k_string.find("\n");
                    }

                    delim = " ";
                    // number of kernel
                    for (int l = 0; l < number_of_kernels; l++){
                        pos = k_string.find(delim);
                        elem = k_string.substr(0,pos);
                        std::string::size_type sz;
                        float elem_f = std::stof(elem, &sz);
                        k_string.erase(0, pos + delim.length());
                        parameters[l](i,j,k) = elem_f;
                    }
                }
            }
        }
        // 

        //
        // Extracting biases
        begin_pos = bias_string.find('['); 
        end_pos = bias_string.find(']');
        bias_string.replace(end_pos, 1, " ");
        bias_string = bias_string.substr(begin_pos + 1, end_pos - begin_pos);

        biases = new float[number_of_kernels];
        delim = " "; pos = 0;
        for (int i = 0; i < number_of_kernels; i++){
            pos = bias_string.find(delim);
            elem = bias_string.substr(0,pos);
            std::string::size_type sz;
            float elem_f = std::stof(elem, &sz);
            bias_string.erase(0, pos + delim.length());
            biases[i] = elem_f;
        }
        //
    }
}

// Method for convolving an image with a kernel
Tensor ConvolutionLayer::convolution(Tensor& image, Tensor& kernel, const size_t& output_size)
{
    Tensor output(output_size, output_size, 1);
    // Convolution
    // iterate over all output pixels of the image
    int image_size = image.get_rows();

    for (int x = 0; x < output_size; x++){
        for (int y = 0; y < output_size; y++){

            // Compute convolution result for each pixel
            float res = 0;
            // iterate over layers
            for (int k = 0; k < kernel.get_layers(); k++){
                // 2D convolution for each layer
                for (int i = 0; i < kernel_size; i++){
                    for (int j = 0; j < kernel_size; j++){
                        res += kernel(i,j,k)*image(x + i,y + j,k);        
                    }
                }
            }

            // Assign convolution result
            output(x,y) = res;
        }
    }
    return output;
}

// Feed forward function of convolution layer taking an input of size 
Tensor ConvolutionLayer::feedForward(Tensor& input)
{
    // Check if the image depth is the same as the kernel depth
    if (input.get_layers() != kernel_depth){
        std::cerr << "Input depth is not equal to the kernel depth!";
        throw("Invalid dimensions!");
    }

    size_t output_size = (input.get_rows() - kernel_size)/stride + 1;
    Tensor layer_output(output_size, output_size, number_of_kernels);

    // Iterating through all kernels 
    for (int nk = 0; nk < number_of_kernels; nk++){

        // 2D output of a single convolution; parameters[nk] is each kernel in the vector of kernels
        Tensor single_output = convolution(input, parameters[nk], output_size);

        // Apply activation function to output of the convolution to create the layer output
        for (int i = 0; i < output_size; i++){
            for (int j = 0; j < output_size; j++){

                if (activation == "None"){
                    layer_output(i,j,nk) = single_output(i,j) + biases[nk];
                } else if (activation == "ReLu"){
                    layer_output(i,j,nk) = std::max(single_output(i,j) + biases[nk],0.0f);
                }
            }
        }
    }

    return layer_output;
}

// Method which fills an array flat_kernels with all kernels
// It essentially flattens out the 4D tensor containing all kernels in this layer
// The flattened array must have size: kernel_size*kernel_size*kernel_depth*number_of_kernels
void ConvolutionLayer::flatten_kernels(float* flat_kernels, size_t flat_kernels_size)
{
    if (flat_kernels_size != kernel_size*kernel_size*kernel_depth*number_of_kernels){
        std::cerr << "Invalid input size for flat kernel array";
        throw("Invalid input size!");
    }

    size_t k_s = kernel_size*kernel_size*kernel_depth;
    float* one_kernel_array = new float[k_s];

    int j = 0;
    // Converting weight parameters into array
    for (auto &v : parameters){
        v.flatten(one_kernel_array, k_s);
        for (int i = 0; i < k_s; i++){
            flat_kernels[k_s*j + i] = one_kernel_array[i];
        }
        j++;
    }

    delete[] one_kernel_array;
}
