#include "PoolingLayer.h"


void PoolingLayer::feedForward()
{
    // Iterating over all layers in input tensor
    for (unsigned int i = 0; i < _inputNumLayers; ++i){
        // Doing max pooling of each layer
        maxPoolLayer(i);
    }
}

void PoolingLayer::maxPoolLayer(unsigned int currentLayerInd)
{
    for (int i = 0; i < _inputNumLayers; ++i) {
        _backPropMapping[i].resize(_outputNumRows*_outputNumCols);
    }

    std::vector<float>        regionOfInterestVals;
    std::vector<unsigned int> regionOfInterestPos;
    regionOfInterestVals.reserve(_blockSize*_blockSize);
    regionOfInterestPos.reserve(_blockSize*_blockSize);

    for (unsigned int i = 0; i < _outputNumRows; ++i) {
        for (unsigned int j = 0; j < _outputNumCols; ++j) {
            for (unsigned int k = 0; k < _blockSize; ++k) {
                for (unsigned int r = 0; r < _blockSize; ++r) {
                    unsigned int row = i*_stride + k;
                    unsigned int clm = j*_stride + r;
                    regionOfInterestVals.push_back(_inputLayers(row, clm, currentLayerInd));
                    regionOfInterestPos.push_back(row + clm*_inputNumCols);
                }
            }
            outputLayers(i, j, currentLayerInd) = *std::max_element(regionOfInterestVals.begin(), regionOfInterestVals.end());
            _backPropMapping[currentLayerInd][i + j*_outputNumCols]
            = regionOfInterestPos[std::max_element(regionOfInterestVals.begin(),regionOfInterestVals.end()) - regionOfInterestVals.begin()];

            regionOfInterestVals.resize(0);
            regionOfInterestPos.resize(0);
        }
    }

}

Tensor PoolingLayer::backPropogation(Tensor& lossGradOfOutput)
{
    Tensor backPropGrad = Tensor(_inputNumRows, _inputNumCols, _inputNumLayers);
    // Iterating over all layers in input tensor
    for (unsigned int i = 0; i < _inputNumLayers; ++i){
        // Doing max unpooling of each layer
        unpool(backPropGrad, lossGradOfOutput, i);
    }
    return backPropGrad;
}

void PoolingLayer::unpool(Tensor& backPropGrad, Tensor& lossGradOfOutput, unsigned int currentLayer)
{
    for (unsigned int j = 0; j < _backPropMapping[currentLayer].size(); ++j) {
        unsigned int fit_i_small = j % _outputNumRows;
        unsigned int fit_j_small = j / _outputNumCols;
        unsigned int fit_i_big   = _backPropMapping[currentLayer][j] % _inputNumRows;
        unsigned int fit_j_big   = _backPropMapping[currentLayer][j] / _inputNumCols;

        backPropGrad(fit_i_big, fit_j_big, currentLayer) += lossGradOfOutput(fit_i_small, fit_j_small, currentLayer);
    }
}

// Testing of that layer:
//std::string image_path = "../python/images_batch_1/airbus_s_000662.png";
//cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
//int rows = 8; int cols = 8; int layers = 2;
//Tensor testLayers(rows, cols, layers);
//float temp = 1.0f;
//for (int k = 0; k < layers; k++){
//for (int i = 0; i < rows; i++){
//for (int j = 0; j < cols; j++){
//testLayers(i,j,k) = temp;
//temp += 1.f;
//}
//}
//}
//std::cout << "Test matrix = " << std::endl;
//std::cout << testLayers << std::endl;
//PoolingLayer poolingLayer(testLayers, 2, 2);
//poolingLayer.feedForward();
//std::cout << "Test matrix after pooling = "  << std::endl;
//std::cout << poolingLayer.outputLayers << std::endl;
//
//Tensor backPropGrad = poolingLayer.outputLayers;
//Tensor backProp = poolingLayer.backPropogation(backPropGrad);
//std::cout << "Back propageted gradient = "  << std::endl;
//std::cout << backProp << std::endl;