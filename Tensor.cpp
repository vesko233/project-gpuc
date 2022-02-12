#include "Tensor.h"

// Parametrized constructor
Tensor::Tensor(size_t number_of_rows, size_t number_of_columns, size_t number_of_layers)
{
	if (number_of_rows < 0 || number_of_columns < 0 || number_of_layers < 0) {
		std::cerr << "The number of rows, columns or layers cannot be a negative number!" ;
		throw("Size is not positive!");
    }
    rows = number_of_rows;
    cols = number_of_columns;
    layers = number_of_layers;
    size = number_of_rows*number_of_columns*number_of_layers;
	tensor_data = new float [size];
	for (size_t i = 0; i < size; i++) {
		tensor_data[i] = 0;
	}    
}

// Copy constructor
Tensor::Tensor(const Tensor& some_tensor)
{
	delete[] this->tensor_data; this->tensor_data = nullptr;
	rows = 0; cols = 0; layers = 0;
	rows = some_tensor.rows; cols = some_tensor.cols; layers = some_tensor.layers; size = some_tensor.size;
    if (some_tensor.size > 0) {
		this->tensor_data = new float[this->size];
		for (size_t i = 0; i < this->size; i++) {
			this->tensor_data[i] = some_tensor.tensor_data[i];
		}
	}
}

// Copy constructor for OpenCV Mat object
Tensor::Tensor(const cv::Mat &image)
{
	delete[] this->tensor_data; this->tensor_data = nullptr;
	rows = image.rows;  cols = image.cols; layers = 3;
    size = rows*cols*layers;
	uchar* image_1D_array = image.isContinuous() ? image.data : image.clone().data;

	if (size > 0) {
		this->tensor_data = new float[size];
		for (size_t i = 0; i < size; i++) {
			this->tensor_data[i] = (float)image_1D_array[i]/255.;
		}
	}
}

// Move constructor
Tensor::Tensor(Tensor&& some_tensor) noexcept
{
	this->rows = some_tensor.rows;
	this->cols = some_tensor.cols;
    this->layers = some_tensor.layers;
	this->size = some_tensor.size;
    this->tensor_data = some_tensor.tensor_data;
	some_tensor.rows = 0; some_tensor.cols = 0; some_tensor.layers = 0; some_tensor.size = 0;
    some_tensor.tensor_data = nullptr;
}

// Return number of rows
size_t Tensor::get_rows() const
{
	return rows;
}

// Returning the number of columns
size_t Tensor::get_cols() const
{
	return cols;
}

// Returning the number of layers
size_t Tensor::get_layers() const
{
	return layers;
}

// Returning the position in Tensor
size_t Tensor::index(size_t n_x, size_t n_y, size_t n_z) const
{
	if (n_x >= 0 && n_x <= rows && n_y >= 0 && n_y <= cols && n_z >= 0 && n_z <= layers) {
        return n_x + n_y*rows + n_z*rows*cols;
	} else {
		std::cerr << "Element out of range! \n";
		exit(1);
	}
}

// Overloading the () operator for a tensor
float& Tensor::operator()(size_t n_x, size_t n_y, size_t n_z) 
{
	return tensor_data[index(n_x,n_y,n_z)];
}


// Copy Assignment operator
Tensor& Tensor::operator=(const Tensor& some_tensor) 
{
	if (&some_tensor == this) {
		return *this;
	}
	delete[] tensor_data; tensor_data = nullptr;
	this->rows = some_tensor.rows; this->cols = some_tensor.cols; this->layers = some_tensor.layers; this->size = some_tensor.size;
	if (this->size > 0) {
		tensor_data = new float[this->size];
		for (size_t i = 0; i < this->size; i++) {
			tensor_data[i] = some_tensor.tensor_data[i];
		}
	}
	return *this;
}

// Move Assignment operator
Tensor& Tensor::operator=(Tensor&& some_tensor) noexcept
{
	std::swap(rows,some_tensor.rows);
	std::swap(cols, some_tensor.cols);
    std::swap(layers, some_tensor.layers);
	std::swap(size, some_tensor.size);
    std::swap(tensor_data, some_tensor.tensor_data);
	return *this;
}
