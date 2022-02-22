cmake_minimum_required(VERSION 3.1)
project(GPU_project)

set(CMAKE_GXX_FLAGS "-Wall -Wextra -Wconversion  -pedantic -std=gnu11")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(main main.cpp Matrix.cpp Tensor.cpp ConvolutionLayer.cpp FullyConnectedLayer.cpp SoftmaxLayer.cpp)
target_link_libraries(main ${OpenCV_LIBS})
