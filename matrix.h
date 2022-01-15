#pragma once
#include<iostream>

// Two functions that I defined in matrix.cpp
bool string_is_float(const std::string& string);
size_t extract_floats(std::string& some_string, std::vector < float >& vector, bool& flag, size_t& row_counter);

// Class of matrices
class matrix
{
private:
	size_t rows{ 0 };
	size_t columns{ 0 };
	float* matrix_data{ nullptr };

public:
	// Default constructor
	matrix() = default;

	// Parametrized constructor
	matrix(size_t number_of_rows, size_t number_of_columns);

	// Copy constructor
	matrix(const matrix& some_matrix);

	// Copy constructor for OpenCV Mat object
	matrix(const cv::Mat &image, const int& channel);

	// Move constructor
	matrix(matrix&& some_matrix) noexcept;

	// Destructor
	~matrix()
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
	matrix& operator=(const matrix& some_matrix);
	matrix& operator=(matrix&& some_matrix) noexcept;

	// Overloading the ostream and istream operators
	friend std::ostream& operator<<(std::ostream& output_stream, const matrix& some_matrix);
	friend std::istream& operator>>(std::istream& input_stream, matrix& some_matrix);

	// Overloading arithmetic operators for matrices
	matrix operator+(const matrix& some_matrix);
	matrix operator-(const matrix& some_matrix);
	matrix operator*(const matrix& some_matrix);

	// Function that finds the determinant of a square matrix
	float calculate_determinant();
};

            
