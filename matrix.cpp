#include<iostream>
#include<string>
#include<sstream>
#include<stdlib.h>
#include<vector>
#include<math.h>
#include "matrix.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// Function that verifies if a string is a float
bool string_is_float(const std::string& string)
{
	char* ptr = 0;
	float value{strtod(string.c_str(), &ptr)};
	return ptr != string.c_str() && *ptr == '\0' && value != HUGE_VAL;
}

// Function that extracts all floats from a line of string
size_t extract_floats(std::string& some_string, std::vector < float >& vector, bool& flag, size_t& row_counter)
{
	std::string whitespaces{" \t\f\v\n\r"};
	size_t find_pos{ some_string.find_last_not_of(whitespaces) };
	if (find_pos != std::string::npos) {
		some_string.erase(find_pos+1);
	} else {
		some_string.clear();
	}
	bool failed{ false };
	std::string element;
	std::stringstream some_stream{ some_string };
	size_t n_counter{ 0 };
	std::vector<float> temp_vec;
	
	while (std::getline(some_stream, element, ' ')) {
		if (string_is_float(element)) {
			std::string::size_type sz;
			temp_vec.push_back(std::stod(element, &sz));
			n_counter++;
		} else {
			failed = true;
			std::cerr << "Invalid input. Value must be a float! \n";
			break;
		}
	}
	if (!failed) {
		for (auto e : temp_vec) {
			vector.push_back(e);
		}
		flag = false;
		row_counter++;
	}
	return n_counter;
}

// ********************************************************************************************************************
// Parametrized constructor
matrix::matrix(size_t number_of_rows, size_t number_of_columns)
{
	std::cout << "Constructing matrix. \n";
	if (number_of_rows < 0 || number_of_columns < 0) {
		std::cerr << "The number of rows or columns cannot be a negative number!" ;
		throw("Size is not positive!");
	}
	rows = number_of_rows;
	columns = number_of_columns;
	matrix_data = new float [number_of_rows * number_of_columns];
	for (size_t i{ 0 }; i < number_of_rows * number_of_columns; i++) {
		matrix_data[i] = 0;
	}
}

// Copy constructor
matrix::matrix(const matrix& some_matrix)
{
	std::cout << "Copy constructor called. \n";
	delete[] this->matrix_data; this->matrix_data = nullptr;
	rows = 0; columns = 0;
	rows = some_matrix.rows; columns = some_matrix.columns;
	if (some_matrix.rows * some_matrix.columns > 0) {
		this->matrix_data = new float[some_matrix.rows * some_matrix.columns];
		for (size_t i{ 0 }; i < some_matrix.rows * some_matrix.columns; i++) {
			this->matrix_data[i] = some_matrix.matrix_data[i];
		}
	}
}

// Copy constructor for OpenCV Mat object
matrix::matrix(const cv::Mat &image, const int& channel)
{
	delete[] this->matrix_data; this->matrix_data = nullptr;
	this->rows = image.rows;  this->columns = image.cols;

	uchar* image_1D_array = image.isContinuous() ? image.data : image.clone().data;

	if (image.rows * image.cols > 0) {
		this->matrix_data = new float[image.rows * image.cols];
		for (size_t i{ 0 }; i < image.rows * image.cols; i++) {
			this->matrix_data[i] = (float)image_1D_array[image.rows*image.cols*channel + i]/255.;
		}
	}
}

// Move constructor
matrix::matrix(matrix&& some_matrix) noexcept
{
	std::cout << "Move constructor called. \n";
	this->rows = some_matrix.rows;
	this->columns = some_matrix.columns;
	this->matrix_data = some_matrix.matrix_data;
	some_matrix.rows = 0; some_matrix.columns = 0;
	some_matrix.matrix_data = nullptr;
}

// Return number of rows
size_t matrix::get_rows() const
{
	return rows;
}

// Returning the number of columns
size_t matrix::get_columns() const
{
	return columns;
}

// Returning the position in matrix
size_t matrix::index(size_t n, size_t m) const
{
	if (n > 0 && n <= rows && m > 0 && m <= columns) {
		return (m - 1) + (n - 1)*columns;
	} else {
		std::cerr << "Element out of range! \n";
		exit(1);
	}
}

// Function that deletes the slected row and column of a matrix
void matrix::delete_row_and_column(size_t n, size_t m)
{
	size_t elements_deleted_from_row{ 0 };
	size_t elements_deleted_from_column{ 0 };
	size_t added_index;
	std::vector<float> temp_vec;
	if (n <= rows && n > 0 && m <= columns && m > 0) {
		for (size_t i{ 0 }; i < rows * columns; i++) {
			temp_vec.push_back(matrix_data[i]);
		}
		for (size_t j{ 1 }; j <= columns; j++) {
			added_index = index(n, j) - elements_deleted_from_row;
			temp_vec.erase(temp_vec.begin() + added_index );
			elements_deleted_from_row++;
		}
		for (size_t k{ 1 }; k <= rows; k++) {
			if (k < n) {
				added_index = index(k, m) - elements_deleted_from_column;
				temp_vec.erase(temp_vec.begin() + added_index );
				elements_deleted_from_column++;
			} else if (k > n) {
				added_index = index(k, m) - elements_deleted_from_row - elements_deleted_from_column;
				temp_vec.erase(temp_vec.begin() + added_index);
				elements_deleted_from_column++;
			}
		}
		rows -= 1; columns -= 1;
		delete[] matrix_data; matrix_data = nullptr;
		matrix_data = new float[rows*columns];
		for (size_t m{ 0 }; m != temp_vec.size(); m++) {
			matrix_data[m] = temp_vec[m];
		}
	} else {
		std::cerr << "The row or column you entered is out of range of the matrix!";
		exit(1);
	}
}

// Overloading the () operator for a matrix
float& matrix::operator()(size_t n, size_t m) 
{
	return matrix_data[index(n, m)];
}

// Copy Assignment operator
matrix& matrix::operator=(const matrix& some_matrix) 
{
	std::cout << "Copy assignment called. \n";
	if (&some_matrix == this) {
		return *this;
	}
	delete[] matrix_data; matrix_data = nullptr;
	rows = some_matrix.rows; columns = some_matrix.columns;
	if (some_matrix.rows*some_matrix.columns > 0) {
		matrix_data = new float[some_matrix.rows * some_matrix.columns];
		for (size_t i{ 0 }; i < some_matrix.rows * some_matrix.columns; i++) {
			matrix_data[i] = some_matrix.matrix_data[i];
		}
	}
	return *this;
}

// Move Assignment operator
matrix& matrix::operator=(matrix&& some_matrix) noexcept
{
	std::cout << "Move assignment called. \n";
	std::swap(rows,some_matrix.rows);
	std::swap(columns, some_matrix.columns);
	std::swap(matrix_data, some_matrix.matrix_data);
	return *this;
}

// Overloading the ostream operator for matrices
std::ostream& operator<<(std::ostream& output_stream, const matrix& some_matrix)
{
	if (some_matrix.rows == 0 && some_matrix.columns == 0) {
		output_stream << "0";
	} else {
		for (size_t n{ 1 }; n <= some_matrix.rows; n++) {
			for (size_t m{ 1 }; m <= some_matrix.columns; m++) {
				size_t i{ some_matrix.index(n,m) };
				output_stream << some_matrix.matrix_data[i] << "  ";
			}
			output_stream << "\n";
		}
	}
	return output_stream;
}

// Overloading the istream operator for matrices
std::istream& operator>>(std::istream& input_stream, matrix& some_matrix)
{
	std::vector<float> extracted_matrix_data;
	size_t columns_n{ 0 }; size_t rows_n{ 0 };
	bool not_finished{ true };
	bool not_finished_first_line{ true };
	std::cout << "Please enter your matrix below. On each row enter the numbers you want in the matrix, separating them with ONE space! \n";
	std::cout << "When you have terminated entering your matrix, type 'x' instead of a row: \n";
	while (not_finished) {
		std::string input_string;
		std::getline(input_stream, input_string);
		if (input_string == "x") {
			not_finished = false;
			break;
		}
		if (not_finished_first_line) {
			columns_n = extract_floats(input_string,extracted_matrix_data, not_finished_first_line,rows_n);
		} else {
			std::string element;
			std::stringstream some_stream{ input_string };
			size_t counter{ 0 };
			while (std::getline(some_stream, element, ' ')) {
				if (string_is_float(element)) {
					counter++;
				} else {
					break;
				}
			}
			if (counter != columns_n) {
				std::cerr << "Invalid input! The entered row is not the same size as the previous one! \n";
			} else {
				size_t n{ 0 };
				bool some_flag{false};
				n = extract_floats(input_string,extracted_matrix_data,some_flag,rows_n);
			}
		}
	}
	some_matrix.matrix_data = nullptr; some_matrix.rows = 0; some_matrix.columns = 0;
	some_matrix.rows = rows_n; some_matrix.columns = columns_n; some_matrix.matrix_data = new float[some_matrix.rows*some_matrix.columns];
	for (size_t i{ 0 }; i != extracted_matrix_data.size(); i++) {
		some_matrix.matrix_data[i] = extracted_matrix_data[i];
	}
	return input_stream;
}

// Overloading the + operator for matrices
matrix matrix::operator+(const matrix& some_matrix)
{
	if (this->rows == some_matrix.rows && this->columns == some_matrix.columns) {
		matrix sum_matrix(some_matrix.rows,some_matrix.columns);
		for (size_t i{ 0 }; i != some_matrix.rows * some_matrix.columns; i++) {
			sum_matrix.matrix_data[i] = this->matrix_data[i] + some_matrix.matrix_data[i];
		}
		return sum_matrix;
	} else {
		throw std::runtime_error("The addition of matrices is possible only if they are the same size!");
	}
}

// Overloading the - operator for matrices
matrix matrix::operator-(const matrix& some_matrix)   
{
	if (this->rows == some_matrix.rows && this->columns == some_matrix.columns) {
		matrix sum_matrix(some_matrix.rows, some_matrix.columns);
		for (size_t i{ 0 }; i != some_matrix.rows * some_matrix.columns; i++) {
			sum_matrix.matrix_data[i] = this->matrix_data[i] - some_matrix.matrix_data[i];
		}
		return sum_matrix;
	}
	else {
		throw std::runtime_error("The subtraction of matrices is possible only if they are the same size!");
	}
}

// Overloading the * operator for matrices
matrix matrix::operator*(const matrix& some_matrix)
{	
	if (this->columns == some_matrix.rows) {
		matrix product_matrix(this->rows,some_matrix.columns);
		for (size_t i{ 1 }; i <= product_matrix.rows; i++) {
			for (size_t j{ 1 }; j <= product_matrix.columns; j++) {
				for (size_t m{ 1 }; m <= some_matrix.rows; m++) {
					product_matrix(i, j) += (this->matrix_data[index(i, m)] * some_matrix.matrix_data[some_matrix.index(m, j)]);
				}	
			}
		}
		return product_matrix;
	} else {
		throw std::runtime_error("The number of columns of the first matrix must be equal to the number of rows of the second one!");
	}
}

// Function that calculates the determinant of a square matrix
float matrix::calculate_determinant()
{
	if (rows == columns) {
		float determinant{ 0 };
		if (columns == 2) {
			determinant += matrix_data[index(1, 1)] * matrix_data[index(2, 2)] - matrix_data[index(2, 1)] * matrix_data[index(1, 2)];
		} else {
			for (size_t i{ 1 }; i <= columns; i++) {
				matrix temp_matrix{ *this };
				temp_matrix.delete_row_and_column(1,i);
				determinant += pow(-1, i + 1) * matrix_data[index(1, i)] * temp_matrix.calculate_determinant();
			}
		}
		return determinant;
	} else {
		throw std::runtime_error("In order to calculate the determinant of a matrix, it must be square.");
	}
}