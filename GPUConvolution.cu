#include "GPUConvolution.cuh"
  
Tensor GPUconvolutuionFeedForward(ConvolutionLayer& conv_layer, Tensor& input)
{
    size_t output_size = (input.get_rows() - conv_layer.kernel_size)/conv_layer.stride + 1;

    // Flattening out kernles
    size_t ker_sizes = conv_layer.kernel_size*conv_layer.kernel_size*conv_layer.kernel_depth*conv_layer.number_of_kernels;
    float* flat_kernels = new float[ker_sizes];
    conv_layer.flatten_kernels(flat_kernels, ker_sizes);

    // Flattening out image
    size_t i_s = input.get_rows()*input.get_cols()*input.get_layers();
    float* input_array = new float[i_s];
    input.flatten(input_array,i_s);

    // Output array
    float* kernel_output = new float[output_size*output_size*conv_layer.number_of_kernels];

    if (conv_layer.number_of_kernels == 16){
        convolution_2D_16(input_array,flat_kernels,kernel_output);
    }else if (conv_layer.number_of_kernels == 32){
        convolution_2D_32(input_array,flat_kernels,kernel_output);
    }

    // Convert output to Tensor
    Tensor layer_output(kernel_output,output_size,output_size,conv_layer.number_of_kernels);

    // Adding biases and activating values
    for (int nk = 0; nk < conv_layer.number_of_kernels; nk++){
        for (int i = 0; i < output_size; i++){
            for (int j = 0; j < output_size; j++){

                if (conv_layer.activation == "None"){
                    layer_output(i,j,nk) += conv_layer.biases[nk];
                } else if (conv_layer.activation == "ReLu"){
                    layer_output(i,j,nk) = std::max(layer_output(i,j,nk) + conv_layer.biases[nk],0.0f);
                }
            }
        }
    }

    // Free memory
    delete[] kernel_output;
    delete[] input_array;
    delete[] flat_kernels;

    return layer_output;
}
