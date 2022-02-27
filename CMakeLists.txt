cmake_minimum_required(VERSION 3.8)
project(GPU_project LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_GXX_FLAGS "-Wall -Wextra -Wconversion  -pedantic -std=g++17 -lstdc++fs")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(main main.cpp Tensor.cpp PoolingLayer.cpp ConvolutionLayer.cpp FullyConnectedLayer.cpp SoftmaxLayer.cpp CudaGEMV.cu CudaConvolution_16.cu CudaConvolution_32.cu GPUConvolution.cu)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main -lcublas -lcurand)

# CUDA Libraries
find_package(CUDA)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS} )
    enable_language(CUDA)
else()
    message("No CUDA found")
endif()



