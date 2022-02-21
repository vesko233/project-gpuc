#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include "Tensor.h"
#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"
#include "SoftmaxLayer.h"


int main()
{
    std::string image_path = "../python/images_batch_1/airbus_s_000662.png";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    Tensor imgTensor = Tensor(img);
    ConvolutionLayer ConLayer1;
    PoolingLayer PolLayer;
    ConvolutionLayer ConLayer2;
    FullyConnectedLayer FCLayer;
    SoftmaxLayer SMLayer;

    ConLayer1 = ConvolutionLayer(5, 3, 1, 10);
    Tensor Conv1 = ConLayer1.feedForward(imgTensor);

    PolLayer = PoolingLayer(Conv1, 2, 2);
    PolLayer.feedForward();

    ConLayer2 = ConvolutionLayer(5, 10, 1, 10);
    Tensor Conv2 = ConLayer2.feedForward(PolLayer.outputLayers);

    float* aaa = new float[10];
    float* bbb = new float[10];
    unsigned int size = Conv2.get_rows()*Conv2.get_cols()*Conv2.get_layers();

    FCLayer = FullyConnectedLayer(10, size, "ReLu");
    FCLayer.feedForward(Conv2.flatten(), aaa, size ,10);

    SMLayer = SoftmaxLayer(10, 10);
    SMLayer.feedForward(aaa, bbb, 10, 10);

    return 0;
}