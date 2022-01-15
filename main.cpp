#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include "matrix.h"

// Defining ReLU activation function
float ReLU(const float &x)
{
    return std::max(x,0.0f);
}

// // Convert image matrix to std::vector 3D array
// std::vector<std::vector<std::vector<float>>> MatToArray(const cv::Mat &image)
// {   
//     // Extracting image matrix to 1D array
//     uchar* image_1D_array = image.isContinuous() ? image.data : image.clone().data ;

//     // Reshaping the flat array to a 3D matrix of floats
//     std::vector<std::vector<std::vector<float>>> image_3D_array(image.rows, std::vector<std::vector<float>>(image.cols, std::vector<float>(image.channels())));
//     for (int i = 0; i < image.rows; i ++){
//         for (int j = 0; j < image.cols; j++){
//             for (int k = 0; k < image.channels(); k++){
//                 image_3D_array[i][j][k] = (float)image_1D_array[(i*image.cols + j)*image.channels() + k]/255.;
//             }
//         }
//     }
//     return image_3D_array;
// }

int main()
{
    std::string image_path = "../../cifar-10/images_batch_1/airbus_s_000662.png";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    return 0;
}