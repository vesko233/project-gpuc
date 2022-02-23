#ifndef GPU_PROJECT_POOLINGLAYER_H
#define GPU_PROJECT_POOLINGLAYER_H

#include "Tensor.h"

class PoolingLayer {
    private:
        unsigned int _stride = 1;
        unsigned int _blockSize = 1;

    public:
        // Constructor
        PoolingLayer(unsigned int stride, unsigned int blockSize);

        // Destructor
        ~PoolingLayer()
        {
            std::cout << std::endl;
        };        

        // Feed forward method
        Tensor feedForward(Tensor& input_tensor);
};

#endif //GPU_PROJECT_POOLINGLAYER_H
        // outputLayers = Tensor(_outputNumRows, _outputNumCols, _inputNumLayers);
