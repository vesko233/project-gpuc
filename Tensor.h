#pragma once
#include<iostream>
#include<string>
#include<sstream>
#include<random>
#include<stdlib.h>
#include<vector>
#include<array>
#include<fstream>
#include<cstdlib>
#include<iterator>
#include<algorithm>
#include<math.h>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

class Tensor
{
    private:
        size_t size{0};
        size_t rows{0};
        size_t cols{0};
        size_t layers{0};
        float* tensor_data{nullptr};        

    public:
	// Default constructor
	Tensor() = default;

	// Parametrized constructor
	Tensor(size_t number_of_rows, size_t number_of_columns, size_t number_of_layers);

	// Copy constructor
	Tensor(const Tensor& some_tensor);

	// Copy constructor for OpenCV Mat object
	Tensor(const cv::Mat &image);

	// Parametrized constructor taking an array of values
	Tensor(float* image, size_t number_of_rows, size_t number_of_columns, size_t number_of_layers);

	// Move constructor
	Tensor(Tensor&& some_tensor) noexcept;

	// Destructor
	~Tensor()
	{
		delete[] tensor_data;
	}

	// Access functions
	size_t get_rows() const;
	size_t get_cols() const;
    size_t get_layers() const;
	size_t index(size_t n_x, size_t n_y, size_t n_z) const;
	float& operator()(size_t n_x, size_t n_y, size_t n_z = 0);

    // Assignment operators
	Tensor& operator=(const Tensor& some_tensor);
	Tensor& operator=(Tensor&& some_tensor) noexcept;

    // Overloading the ostream operator
    friend std::ostream& operator<<(std::ostream& output_stream, const Tensor& some_Tensor);

	// Arithmetic operators and functions
	Tensor operator*(const Tensor& some_tensor);
	Tensor hadamard(const Tensor& some_tensor);
	Tensor transpose();
	void flatten(float* flat, const size_t& flat_size);
};