#pragma once
#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// Two functions that I defined in Matrix.cpp
bool string_is_float(const std::string& string);
size_t extract_floats(std::string& some_string, std::vector < float >& vector, bool& flag, size_t& row_counter);

// Class of matrices
class Matrix
{
private:
	size_t rows{ 0 };
	size_t columns{ 0 };
	float* matrix_data{ nullptr };

public:
	// Default constructor
	Matrix() = default;

	// Parametrized constructor
	Matrix(size_t number_of_rows, size_t number_of_columns);

	// Copy constructor
	Matrix(const Matrix& some_matrix);

	// Copy constructor for OpenCV Mat object
	Matrix(const cv::Mat &image, const int& channel);

	// Move constructor
	Matrix(Matrix&& some_matrix) noexcept;

	// Destructor
	~Matrix()
	{
		delete[] matrix_data;
		std::cout << "Destructor called. Deleting matrix." << std::endl;
	}

	// Access functions
	size_t get_rows() const;
	size_t get_columns() const;
	size_t index(size_t m, size_t n) const;
	float& operator()(size_t n, size_t m);
	void delete_row_and_column(size_t n,size_t m);

	// Assignment operators
	Matrix& operator=(const Matrix& some_matrix);
	Matrix& operator=(Matrix&& some_matrix) noexcept;

	// Overloading the ostream and istream operators
	friend std::ostream& operator<<(std::ostream& output_stream, const Matrix& some_matrix);
	friend std::istream& operator>>(std::istream& input_stream, Matrix& some_matrix);

	// Overloading arithmetic operators for matrices
	Matrix operator+(const Matrix& some_matrix);
	Matrix operator-(const Matrix& some_matrix);
	Matrix operator*(const Matrix& some_matrix);

	// Function that finds the determinant of a square matrix
	float calculate_determinant();
};

            
