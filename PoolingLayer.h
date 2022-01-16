#ifndef GPU_PROJECT_POOLINGLAYER_H
#define GPU_PROJECT_POOLINGLAYER_H

#include <iostream>
#include <string>
#include <sstream>
#include "Matrix.h"

class PoolingLayer {

public:
    PoolingLayer(const std::vector<Matrix>& inputLayers, unsigned int stride, unsigned int blockSize):
        _inputLayers(inputLayers), _stride(stride), _blockSize(blockSize)
    {
        outputLayers.reserve(inputLayers.size());
        _backPropMapping.resize(_inputLayers.size());

        std::cout << "Pooling parameters: stride = "  << stride
                  << ", size of the pooling block = " << blockSize << std::endl;
    };

    ~PoolingLayer()
    {
        std::cout << std::endl;
    };

    std::vector<Matrix> outputLayers;
    void feedForward();
    std::vector<Matrix> backPropogation(const std::vector<Matrix>& lossGradOfOutput);
    Matrix unpool(unsigned int i, Matrix& lossGradOfOutput);

private:
    unsigned int _stride;
    unsigned int _blockSize;
    std::vector<Matrix> _inputLayers;
    std::vector<std::vector<unsigned int>> _backPropMapping;

    Matrix maxPool(Matrix& toBePooled, std::vector<unsigned int>& backPropMapping);
};

#endif //GPU_PROJECT_POOLINGLAYER_H
