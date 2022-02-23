#include "PoolingLayer.h"

PoolingLayer::PoolingLayer(unsigned int stride, unsigned int blockSize)
{
    if (stride < 1 || blockSize < 1){
        std::cerr << "Cannot have stride and/or block size smaller than 1!";
        throw("Invalid parameters");
    }
    _stride = stride;
    _blockSize = blockSize;
}            


// Feed forward method for pooling layer. 
Tensor PoolingLayer::feedForward(Tensor& input_tensor)
{
    if (input_tensor.get_rows() != input_tensor.get_cols()){
        std::cerr << "Rows and columns of Maxpool layer should be the same!";
        throw("Invalid input parameters!");
    }

    // Creating output tensor
    size_t output_size = input_tensor.get_rows()/_stride - (_blockSize - _stride);
    Tensor output_tensor(output_size,output_size,input_tensor.get_layers());

    // Iterate through layers
    for (int m = 0; m < input_tensor.get_layers(); m++){

        //Iterate through all output elements
        for (int i = 0; i < output_size; i++){
            for (int j = 0; j < output_size; j++){
                
                size_t temp_array_size = _blockSize*_blockSize;
                float temp_array[temp_array_size];

                // Iterate through block
                for (int k = 0; k < _blockSize; k++){
                    for (int l = 0; l < _blockSize; l++){
                        temp_array[k + l*_blockSize] = input_tensor(i*_stride + k,j*_stride + l,m);
                    }
                }
                float res = *std::max_element(temp_array, temp_array + temp_array_size);
                output_tensor(i,j,m) = res;
            }
        }
    }

    return output_tensor;
}