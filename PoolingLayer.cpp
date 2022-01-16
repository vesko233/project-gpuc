#include "PoolingLayer.h"


void PoolingLayer::feedForward()
{
    for (unsigned int i = 0; i < _inputLayers.size(); ++i){
        outputLayers.push_back(maxPool(_inputLayers[i], _backPropMapping[i]));
    }
}

Matrix PoolingLayer::maxPool(Matrix &toBePooled, std::vector<unsigned int>& backPropMapping)
{
    unsigned int number_of_rows = toBePooled.get_rows()   /_stride - (_blockSize - _stride);
    unsigned int number_of_cols = toBePooled.get_columns()/_stride - (_blockSize - _stride);
    for (int i = 0; i < _inputLayers.size(); ++i) {
        _backPropMapping[i].resize(number_of_rows*number_of_cols);
    }

    Matrix pooledMat(number_of_rows, number_of_cols);
    std::vector<float>        regionOfInterestVals;
    std::vector<unsigned int> regionOfInterestPos;
    regionOfInterestVals.reserve(_blockSize*_blockSize);
    regionOfInterestPos.reserve(_blockSize*_blockSize);

    for (unsigned int i = 1; i <= number_of_rows; ++i) {
        for (unsigned int j = 1; j <= number_of_cols; ++j) {
            for (unsigned int k = 1; k <= _blockSize; ++k) {
                for (unsigned int r = 1; r <= _blockSize; ++r) {
                    unsigned int row = (i-1)*_stride + k;
                    unsigned int clm = (j-1)*_stride + r;
                    regionOfInterestVals.push_back(toBePooled(row, clm));
                    regionOfInterestPos.push_back((row - 1) + (clm-1)*toBePooled.get_columns());
                }
            }
            pooledMat(i, j) = *std::max_element(regionOfInterestVals.begin(), regionOfInterestVals.end());
            backPropMapping[(i - 1) + (j - 1)*number_of_cols] = regionOfInterestPos[std::max_element(regionOfInterestVals.begin(),regionOfInterestVals.end()) - regionOfInterestVals.begin()];

            regionOfInterestVals.resize(0);
            regionOfInterestPos.resize(0);
        }
    }

    return pooledMat;
}

std::vector<Matrix> PoolingLayer::backPropogation(const std::vector<Matrix>& lossGradOfOutput)
{
    std::vector<Matrix> backPropGrad;
    backPropGrad.reserve(_inputLayers.size());
    for (int i = 0; i < _inputLayers.size(); ++i) {
        backPropGrad.push_back(unpool(i, const_cast<Matrix &>(lossGradOfOutput[i])));
    }
    return backPropGrad;
}

Matrix PoolingLayer::unpool(unsigned int i, Matrix& lossGradOfOutput)
{
    Matrix unpooled(_inputLayers[i].get_rows(), _inputLayers[i].get_columns());
    for (unsigned int j = 0; j < _backPropMapping[i].size(); ++j) {
        unsigned int fit_i_small = j % outputLayers[i].get_columns() + 1;
        unsigned int fit_j_small = j / outputLayers[i].get_columns() + 1;
        unsigned int fit_i_big   = _backPropMapping[i][j] % _inputLayers[i].get_columns() + 1;
        unsigned int fit_j_big   = _backPropMapping[i][j] / _inputLayers[i].get_columns() + 1;

        unpooled(fit_i_big, fit_j_big) += lossGradOfOutput(fit_i_small, fit_j_small);
    }
    return unpooled;
}

// Testing of that layer:
//std::vector<Matrix> testLayers = setTestData(2, 8, 8);
//std::cout << "Test matrix = " << std::endl;
//std::cout << testLayers[1] << std::endl;
//PoolingLayer poolingLayer(testLayers, 2, 2);
//poolingLayer.feedForward();
//std::cout << "Test matrix after pooling = "  << std::endl;
//std::cout << poolingLayer.outputLayers[1] << std::endl;
//
//std::vector<Matrix> backPropGrad = setTestData(2, 4, 4);
//std::vector<Matrix> backProp = poolingLayer.backPropogation(backPropGrad);
//std::cout << "Back propageted gradient = "  << std::endl;
//std::cout << backProp[1] << std::endl;