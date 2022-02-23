#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <vector>
#include "Tensor.h"
#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"
#include "SoftmaxLayer.h"


int main()
{
    // Defining architecture
    ConvolutionLayer convLayer1(3,3,1,16,"ReLu","../cnn-weights/cnn-weights-conv2d.txt"); 
    std::cout << "hey1" << std::endl;
    // Output shape: (30, 30, 16)

    ConvolutionLayer convLayer2(3,16,1,32,"ReLu","../cnn-weights/cnn-weights-conv2d_1.txt");
    // Output shape: (28, 28, 32)
    std::cout << "hey2" << std::endl;

    PoolingLayer poolLayer(2,2);
    // Output shape: (14, 14, 32)

    // Flatten last output to (6272)
    FullyConnectedLayer denseLayer(256,6272,"ReLu","../cnn-weights/cnn-weights-dense.txt");
    // Output shape: (256)
    std::cout << "hey3" << std::endl;

    SoftmaxLayer denseSoftmaxLayer(10,256,"../cnn-weights/cnn-weights-dense_1.txt");
    // Output shape: (10)
    std::cout << "hey4" << std::endl;


    // Allocating memory for needed arrays during feed forward of neural network
    float* output_flat = new float[6272];
    float* dense_layer_output = new float[256];
    float* softamx_layer_output = new float[10];

    // Allocating memory for array containing class predictions
    int* predicted_classes = new int[10000];
    int* true_classes = new int[10000];

    // Allocate memory for image array conainer to read from csv file
    float* image_array = new float[3*1024];

    // image counter
    int counter = 0;

    std::ifstream testSetFile("../../cifar-10/test.csv");
    if (testSetFile.is_open()){
        // Getting entire string of file
        std::string str;
        std::getline(testSetFile,str); // Skip first line

        // Getting each line of file
        while(std::getline(testSetFile,str)){
            std::istringstream input_image_str(str);
            std::string elem;
            float elem_f;

            // Extracting each image from csv file and storing it in image_array
            int i = 0;
            while (std::getline(input_image_str, elem, ',')){
                std::string::size_type sz;
                elem_f = std::stof(elem, &sz);
                image_array[i] = elem_f;
                i++;
            }

            // Making a Tensor object from extracted image array
            Tensor input_image(image_array,32,32,3);

            // pass to 1st conv layer
            Tensor output = convLayer1.feedForward(input_image);
            
            // pass to 2nd conv layer
            output = convLayer2.feedForward(output);

            // pass to maxpool layer
            output = poolLayer.feedForward(output);

            // Flatten output
            output_flat = output.flatten(output_flat,6272);

            // pass to dense fully connected layer
            dense_layer_output = denseLayer.feedForward(output_flat,dense_layer_output,6272,256);

            // pass to softmax layer
            softamx_layer_output = denseSoftmaxLayer.feedForward(dense_layer_output,softamx_layer_output,256,10);

            std::cout << "image " << counter + 1 << " classified." << std::endl;

            predicted_classes[counter] = std::max_element(softamx_layer_output,softamx_layer_output+10) - softamx_layer_output;
            counter++;
        }
    }

    // Fetching true labels from csv
    std::ifstream testLabels("../cnn-weights/Y-test.csv");
    if (testLabels.is_open()){
        std::string str;
        while(std::getline(testLabels,str)){
            std::istringstream input_line_str(str);
            std::string elem;
            int elem_i; 

            int i = 0;
            while (std::getline(input_line_str, elem, ',')){
                std::string::size_type sz;
                elem_i = std::stoi(elem, &sz);
                true_classes[i] = elem_i;
                i++;
            }
        }
    }

    float accuracy = 0;
    for (int i = 0; i < 10000; i++){
        if (predicted_classes[i] == true_classes[i]) accuracy++;
    }
    accuracy  = accuracy/10000;

    std::cout << "Accuracy for test set = " << accuracy*100 << "%" << std::endl;

    // Free memory
    delete[] predicted_classes;
    delete[] true_classes;
    delete[] image_array;
    delete[] output_flat;
    delete[] dense_layer_output;
    delete[] softamx_layer_output;
}







    // ConvolutionLayer ConLayer1;
    // PoolingLayer PolLayer;
    // ConvolutionLayer ConLayer2;
    // FullyConnectedLayer FCLayer;
    // SoftmaxLayer SMLayer;

    // ConLayer1 = ConvolutionLayer(5, 3, 1, 10);
    // Tensor Conv1 = ConLayer1.feedForward(imgTensor);

    // PolLayer = PoolingLayer(Conv1, 2, 2);
    // PolLayer.feedForward();

    // ConLayer2 = ConvolutionLayer(5, 10, 1, 10);
    // Tensor Conv2 = ConLayer2.feedForward(PolLayer.outputLayers);

    // float* aaa = new float[10];
    // float* bbb = new float[10];
    // unsigned int size = Conv2.get_rows()*Conv2.get_cols()*Conv2.get_layers();

    // FCLayer = FullyConnectedLayer(10, size, "ReLu");
    // FCLayer.feedForward(Conv2.flatten(), aaa, size ,10);

    // SMLayer = SoftmaxLayer(10, 10);
    // SMLayer.feedForward(aaa, bbb, 10, 10);

    // return 0;

    ///////////////////////////////////////////////////







    // // Getting a test image
    // std::string image_path = "../../cifar-10/images_batch_2/american_elk_s_000918.png";
    // cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    // Tensor test_img = Tensor(img);