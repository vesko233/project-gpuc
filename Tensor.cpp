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

// Parametrized constructor, taking data from an array
Tensor::Tensor(float* image, size_t number_of_rows, size_t number_of_columns, size_t number_of_layers)
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
		tensor_data[i] = image[i]/255.;
	}    
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
	if (n_x >= 0 && n_x < rows && n_y >= 0 && n_y < cols && n_z >= 0 && n_z < layers) {
        return n_y + n_x*cols + n_z*rows*cols;
	} else {
		std::cerr << "Element out of range! \n";
		std::cerr << "At position: n_x = " << n_x << ", n_y = " << n_y << ", n_z = " << n_z << std::endl;
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

// Overloading the ostream operator
std::ostream &operator<<(std::ostream &output_stream, const Tensor &some_Tensor)
{
        if (some_Tensor.rows == 0 && some_Tensor.cols == 0) {
            output_stream << "0";
        } else {
            for (size_t k{0}; k < some_Tensor.layers; k++) {
                output_stream << "Layer number : " << k << std::endl;
                for (size_t n{0}; n < some_Tensor.rows; n++) {
                    for (size_t m{0}; m < some_Tensor.cols; m++) {
                        size_t i{some_Tensor.index(n, m, k)};
                        output_stream << some_Tensor.tensor_data[i] << "  ";
                    }
                    output_stream << "\n";
                }
            }
        }
        return output_stream;
}

// Multiplication operator. Works only for tensors with 1 layer, i.e. matrices. It essentially performs matrix multiplication.
Tensor Tensor::operator*(const Tensor& some_tensor)
{
	if (this->layers != 1 || some_tensor.layers != 1){
		std::cerr << "Tensor multiplication is overloaded only for 2D tensors, i.e. for tensors with only 1 layer";
		throw("Invalid tensor size!");
	}
	if (this->cols == some_tensor.rows) {
		Tensor product_tensor(this->rows,some_tensor.cols,1);
		for (int i = 0; i < product_tensor.rows; i++) {
			for (int j = 0; j < product_tensor.cols; j++) {
				for (int m = 0; m < some_tensor.rows; m++) {
					product_tensor(i, j) += this->tensor_data[index(i,m,0)]*some_tensor.tensor_data[index(m,j,0)];
				}	
			}
		}
		return product_tensor;
	} else {
		throw std::runtime_error("The number of columns of the first tensor must be equal to the number of rows of the second one!");
	}
}

// Hadamard product between two tensors
Tensor Tensor::hadamard(const Tensor& some_tensor)
{
	if (this->rows != some_tensor.rows || this->cols != some_tensor.cols || this->layers != some_tensor.layers){
		std::cerr << "Hadamard product can only be performed on tensors with the same dimensions!";
		throw("Invalid dimensions!");
	}
	Tensor hadamard_tensor(this->rows, this->cols, this->layers);
	for (int i = 0; i < this->rows; i++){
		for (int j = 0; j < this->cols; j++){
			for (int k = 0; k < this->layers; k++){
				hadamard_tensor(i,j,k) = this->tensor_data[index(i,j,k)]*some_tensor.tensor_data[index(i,j,k)];
			}	
		}
	}
	return hadamard_tensor;
}

// Transpose of a tensor. Works only for 2D tensors, i.e. matrices
Tensor Tensor::transpose()
{
	if (layers != 1){
		std::cerr << "The transpose method works only for 2D tensors!";
		throw("Invalid dimension");
	}
	Tensor transposed_tensor(this->cols, this->rows, this->layers);
	// Exchange rows and columns
	for (int i = 0; i < this->cols; i++){
		for (int j = 0; j < this->rows; j++){
			transposed_tensor(i,j) = this->tensor_data[index(j,i,0)];
		}
	}
	return transposed_tensor;
} 

// Flatten tensor(Return tensor data)
void Tensor::flatten(float* flat, const size_t& flat_size)
{
	if (flat_size != size){
		std::cerr << "Output array for flattened tensor is different than size of Tensor data!";
		throw("Invalid input!");
	}
	for (int i = 0; i < flat_size; i++){
		flat[i] = tensor_data[i];
	}
}