#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <chrono>
#include "Tensor.h"
#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxLayer.h"
#include "FullyConnectedLayer.h"
#include "GPUConvolution.cuh"
#include <numeric>

double calculateMean(std::vector<double> &vec){
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

int main()
{
    bool useGPU = false;
    std::string path;
    if(useGPU) path = "./cnn-weights-new/";
    else path = "./cnn-weights-new/";
    // If exit_image == -1 it means we run network on all test data, else - on the chosen part of it
    int exit_image = -1;

    // Defining CNN architecture
    //==================
    // First convolutional layer
    ConvolutionLayer convLayer1(3,3,1,16,"ReLu",path + "cnn-weights-conv2d.txt");
    // Output shape: (30, 30, 16)

    // Second convoloutional layer
    ConvolutionLayer convLayer2(3,16,1,32,"ReLu",path + "cnn-weights-conv2d_1.txt");
    // Output shape: (28, 28, 32)

    // Max pooling layer
    PoolingLayer poolLayer(2,2);
    // Output shape: (14, 14, 32)

    // Flatten last output to (6272) = 14*14*32

    // Dense layer
    FullyConnectedLayer denseLayer(256,6272,"ReLu",path + "cnn-weights-dense.txt");
    // Output shape: (256)

    // Final dense softmax layer
    SoftmaxLayer denseSoftmaxLayer(10,256,path + "cnn-weights-dense_1.txt");
    // Output shape: (10)
    //==================


    // Memory allocation
    //==================
    // Allocate memory for image array conainer to read from csv file
    float* image_array = new float[3*1024];

    // Allocating memory for needed arrays during feed forward of neural network
    float* output_flat = new float[6272];

    // Allocating memory for dense layer output
    float* dense_layer_output = new float[256];
    float* dense_layer_output_activated = new float[256];
    
    // Allocating memory for final softmax layer
    float* softamx_layer_output = new float[10];
    float* softamx_layer_output_activated = new float[10];

    // Allocating memory for array containing class predictions
    int* predicted_classes = new int[10000];
    int* true_classes = new int[10000];
    //==================

    int counter = 0;

    Tensor conv_layer1_output;
    Tensor conv_layer2_output;
    Tensor maxpool_layer_output;

    // Open csv file with test data. The CNN will attempt to classify 10 000 32x32x3 images
    std::ifstream testSetFile(path + "X-test.csv");

    // Timers for whole run of CNN
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    std::vector<double> times_per_image;
    std::vector<double> times_per_conv1;
    std::vector<double> times_per_conv2;
    std::vector<double> times_per_gemv1;
    std::vector<double> times_per_gemv2;
    times_per_image.reserve(10000);
    times_per_conv1.reserve(10000);
    times_per_conv2.reserve(10000);
    times_per_gemv1.reserve(10000);
    times_per_gemv2.reserve(10000);

    if (testSetFile.is_open()){
        // Getting entire string of file
        std::string str;

        // Getting each line of file
        while(std::getline(testSetFile,str)){
            std::istringstream input_image_str(str);
            std::string elem;
            float elem_f;

            // Extracting each image from csv file and storing it in image_array. Initializing Tensor object for image.
            int i = 0;
            while (std::getline(input_image_str, elem, ',')){
                std::string::size_type sz;
                elem_f = std::stof(elem, &sz);
                image_array[i] = elem_f/255;
                i++;
            }
            Tensor input_image(image_array,32,32,3);


            std::chrono::time_point<std::chrono::system_clock> start_per_image, end_per_image;
            std::chrono::time_point<std::chrono::system_clock> start_per_layer, end_per_layer;
            std::chrono::duration<double> elapsed_seconds_per_layer;
            start_per_image = std::chrono::system_clock::now();
            // Cuda code for GPU(parallel)
            if (useGPU){
                // Pass image through first convolution layer

                start_per_layer = std::chrono::system_clock::now();
                conv_layer1_output = GPUconvolutuionFeedForward(convLayer1, input_image);
                end_per_layer = std::chrono::system_clock::now();
                elapsed_seconds_per_layer = end_per_layer-start_per_layer;
                times_per_conv1.push_back(elapsed_seconds_per_layer.count());

                // Pass to second convolution layer
                start_per_layer = std::chrono::system_clock::now();
                conv_layer2_output = GPUconvolutuionFeedForward(convLayer2, conv_layer1_output);
                end_per_layer = std::chrono::system_clock::now();
                elapsed_seconds_per_layer = end_per_layer-start_per_layer;
                times_per_conv2.push_back(elapsed_seconds_per_layer.count());


                // Pass to maxpool layer
                maxpool_layer_output = poolLayer.feedForward(conv_layer2_output);

                // Flatten output
                maxpool_layer_output.flatten(output_flat,6272);

                // Pass to dense fully connected layer
                start_per_layer = std::chrono::system_clock::now();
                denseLayer.feedForward(output_flat,dense_layer_output,6272,256,useGPU);
                end_per_layer = std::chrono::system_clock::now();
                elapsed_seconds_per_layer = end_per_layer-start_per_layer;
                times_per_gemv1.push_back(elapsed_seconds_per_layer.count());
                denseLayer.activate(dense_layer_output,dense_layer_output_activated,256,256);

                // Pass to softmax layer
                start_per_layer = std::chrono::system_clock::now();
                denseSoftmaxLayer.feedForward(dense_layer_output,softamx_layer_output,256,10,useGPU);
                end_per_layer = std::chrono::system_clock::now();
                elapsed_seconds_per_layer = end_per_layer-start_per_layer;
                times_per_gemv2.push_back(elapsed_seconds_per_layer.count());
                denseSoftmaxLayer.softmaxActivate(softamx_layer_output,softamx_layer_output_activated,10,10);

            // Code for CPU(serial)
            }else{
                // Pass image through first convolution layer
                start_per_layer = std::chrono::system_clock::now();
                conv_layer1_output = convLayer1.feedForward(input_image);
                end_per_layer = std::chrono::system_clock::now();
                elapsed_seconds_per_layer = end_per_layer-start_per_layer;
                times_per_conv1.push_back(elapsed_seconds_per_layer.count());

                // Pass to second convolution layer
                start_per_layer = std::chrono::system_clock::now();
                conv_layer2_output = convLayer2.feedForward(conv_layer1_output);
                end_per_layer = std::chrono::system_clock::now();
                elapsed_seconds_per_layer = end_per_layer-start_per_layer;
                times_per_conv2.push_back(elapsed_seconds_per_layer.count());

                // Pass to maxpool layer
                maxpool_layer_output = poolLayer.feedForward(conv_layer2_output);

                // Flatten output
                maxpool_layer_output.flatten(output_flat,6272);

                // Pass to dense fully connected layer
                start_per_layer = std::chrono::system_clock::now();
                denseLayer.feedForward(output_flat,dense_layer_output,6272,256,useGPU);
                end_per_layer = std::chrono::system_clock::now();
                elapsed_seconds_per_layer = end_per_layer-start_per_layer;
                times_per_gemv1.push_back(elapsed_seconds_per_layer.count());
                denseLayer.activate(dense_layer_output,dense_layer_output_activated,256,256);

                // Pass to softmax layer
                start_per_layer = std::chrono::system_clock::now();
                denseSoftmaxLayer.feedForward(dense_layer_output,softamx_layer_output,256,10,useGPU);
                end_per_layer = std::chrono::system_clock::now();
                elapsed_seconds_per_layer = end_per_layer-start_per_layer;
                times_per_gemv2.push_back(elapsed_seconds_per_layer.count());
                denseSoftmaxLayer.softmaxActivate(softamx_layer_output,softamx_layer_output_activated,10,10);
            }
            end_per_image = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds_per_image = end_per_image-start_per_image;
            times_per_image.push_back(elapsed_seconds_per_image.count());

            for (int i = 0; i < 10; i++){
                std::cout << softamx_layer_output_activated[i] << "; ";
            }
            std::cout << std::endl;

            std::cout << "image " << counter + 1 << " classified as ";

            predicted_classes[counter] = std::max_element(softamx_layer_output_activated,softamx_layer_output_activated+10) - softamx_layer_output_activated;
            std::cout << predicted_classes[counter] << std::endl;
            counter++;

            if (counter == exit_image){
                std::cout << "Average time per image: " << calculateMean(times_per_image) << "s\n";
                std::cout << "Average time per conv1: " << calculateMean(times_per_conv1) << "s\n";
                std::cout << "Average time per conv2: " << calculateMean(times_per_conv2) << "s\n";
                std::cout << "Average time per gemv1: " << calculateMean(times_per_gemv1) << "s\n";
                std::cout << "Average time per gemv2: " << calculateMean(times_per_gemv2) << "s\n";
                exit(exit_image);
            }


            // for (int i = 0; i < output2_C.get_rows(); i++){
            //     for (int j = 0; j < output2_C.get_cols(); j++){
            //         for (int k = 0; k < output2_C.get_layers(); k++){
            //             std::cout << output2_nC(i,j,k) << " | " << output2_C(i,j,k) << std::endl;
            //         }
            //     }
            // }
        }
    }

    // Fetching true labels from csv
    std::ifstream testLabels(path + "Y-test.csv");
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

    // Computing accuracy of classified images
    float accuracy = 0;
    for (int i = 0; i < 10000; i++){
        if (predicted_classes[i] == true_classes[i]) accuracy++;
    }
    accuracy  = accuracy/10000;
    std::cout << "Accuracy for test set = " << accuracy*100 << "%" << std::endl;

    // Measure the times
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "Average time per image: " << calculateMean(times_per_image) << "s\n";
    std::cout << "Average time per conv1: " << calculateMean(times_per_conv1) << "s\n";
    std::cout << "Average time per conv2: " << calculateMean(times_per_conv2) << "s\n";
    std::cout << "Average time per gemv1: " << calculateMean(times_per_gemv1) << "s\n";
    std::cout << "Average time per gemv2: " << calculateMean(times_per_gemv2) << "s\n";
    std::cout << "Total elapsed time:     " << elapsed_seconds.count() << "s\n";


    // Freeing allocated memory
    delete[] image_array;
    delete[] output_flat;
    delete[] dense_layer_output;
    delete[] dense_layer_output_activated;
    delete[] softamx_layer_output;
    delete[] softamx_layer_output_activated;
    delete[] predicted_classes;
    delete[] true_classes;
}
