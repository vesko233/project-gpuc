#ifndef GPU_PROJECT_POOLINGLAYER_H
#define GPU_PROJECT_POOLINGLAYER_H

#include "Tensor.h"

class PoolingLayer {

public:
    PoolingLayer(const Tensor& inputLayers, unsigned int stride, unsigned int blockSize):
        _inputLayers(inputLayers), _stride(stride), _blockSize(blockSize)
    {
        // Getting dimensions of input tensor
        _inputNumLayers = inputLayers.get_layers();
        _inputNumRows = inputLayers.get_rows();
        _inputNumCols = inputLayers.get_cols();
        // Calculating dimensions of output tensor
        _outputNumRows = _inputNumRows/_stride - (_blockSize - _stride);
        _outputNumCols = _inputNumCols/_stride - (_blockSize - _stride);
        outputLayers = Tensor(_outputNumRows, _outputNumCols, _inputNumLayers);
        _backPropMapping.resize(_inputNumLayers);

        std::cout << "Pooling parameters: stride = "  << stride
                  << ", size of the pooling block = " << blockSize << std::endl;
    };

    ~PoolingLayer()
    {
        std::cout << std::endl;
    };

    Tensor outputLayers;
    void feedForward();
    Tensor backPropogation(Tensor& lossGradOfOutput);


private:
    unsigned int _stride{};
    unsigned int _blockSize{};
    unsigned int _inputNumLayers{};
    unsigned int _inputNumRows{};
    unsigned int _inputNumCols{};
    unsigned int _outputNumRows{};
    unsigned int _outputNumCols{};
    Tensor _inputLayers;
    std::vector<std::vector<unsigned int>> _backPropMapping;

    void maxPoolLayer(unsigned int currentLayerInd);
    void unpool(Tensor& backPropGrad, Tensor& lossGradOfOutput, unsigned int currentLayer);
};

#endif //GPU_PROJECT_POOLINGLAYER_H
