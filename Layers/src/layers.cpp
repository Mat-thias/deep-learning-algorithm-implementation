#include "layers.h"

/**
 * @brief Virtual base class method for forward pass.
 * 
 * This is a placeholder for derived classes to override.
 * 
 * @param input Pointer to input data.
 * @param output Pointer to output data.
 */
void Layer::forward(float *input, float *output) {
    // Base class: does nothing by default
}

/**
 * @brief Constructs a FullyConnected layer with given dimensions and parameters.
 * 
 * @param input_size Number of input nodes.
 * @param output_size Number of output nodes.
 * @param weights Pointer to the weight array (flattened 2D: input_size x output_size).
 * @param bias Pointer to the bias array of size output_size.
 */
FullyConnected::FullyConnected(uint32_t input_size, uint32_t output_size, float *weights, float *bias) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->weights = weights;
    this->bias = bias;
}

/**
 * @brief Performs the forward pass of a FullyConnected layer.
 * 
 * Computes: output[j] = sum(input[i] * weights[i][j]) + bias[j]
 * 
 * @param input Pointer to input data array of size input_size.
 * @param output Pointer to output data array of size output_size.
 */
void FullyConnected::forward(float *input, float *output) {
    for (uint32_t j = 0; j < this->output_size; j++) {
        output[j] = 0;
        for (uint32_t i = 0; i < this->input_size; i++) {
            output[j] += input[i] * this->weights[(i * this->output_size) + j];
        }
        output[j] += this->bias[j];
    }
}

/**
 * @brief Constructs a ReLU activation layer.
 * 
 * @param input_dim Number of dimensions in the input.
 * @param input_shape Pointer to array representing the shape of the input.
 */
Relu::Relu(uint32_t input_dim, uint32_t *input_shape) {
    this->input_dim = input_dim;
    this->input_shape = input_shape;
}

/**
 * @brief Performs the ReLU activation on the input.
 * 
 * Applies ReLU: output[i] = max(0, input[i])
 * 
 * @param input Pointer to input data array.
 * @param output Pointer to output data array.
 */
void Relu::forward(float *input, float *output) {
    int input_size = 0;

    for (uint32_t i = 0; i < this->input_dim; i++) {
        input_size += this->input_shape[i];
    }

    for (uint32_t i = 0; i < input_size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}



/**
 * @brief Constructs a Convolutional2DLayer with given dimensions and parameters.
 * 
 * @param input_row_size Number of rows in the input.
 * @param input_col_size Number of columns in the input.
 * @param output_row_size Number of rows in the output.
 * @param output_col_size Number of columns in the output.
 * @param kernels Pointer to the flattened kernel weights array.
 * @param bias Pointer to the bias array.
 */
Convolutional2DLayer::Convolutional2DLayer(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size, 
                                            uint32_t output_channel_size, int32_t kernel_row_size, uint32_t kernel_col_size, 
                                            uint32_t stride_row, uint32_t stride_col, uint32_t padding,
                                            float *kernels, float *bias) {
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->padding = padding;
    
    this->kernels = kernels;
    this->bias = bias;

    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;
}

/**
* @brief Performs the forward pass of a 2D convolution layer.
* 
* Note: This is a placeholder function. Actual implementation is required.
* 
* @param input Pointer to the input data array.
* @param output Pointer to the output data array.
*/
void Convolutional2DLayer::forward(float *input, float *output) {
    
    int output_index, input_index, kernel_index, intermidate;

    switch (this->padding) {
    case PADDING_VALID:

        for (uint32_t n=0; n<this->output_channel_size; n++) {
            for (uint32_t m=0; m<this->output_row_size; m++) {
                for (uint32_t l=0; l<this->output_col_size; l++) {

                    output[
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l] = 0;

                    for (uint32_t k=0; k<input_channel_size; k++) {
                        for (uint32_t j=0; j<kernel_row_size; j++) {
                            for (uint32_t i=0; i<kernel_col_size; i++) {
                                output_index = (n * this->output_row_size * this->output_col_size) + 
                                                (m * this->output_col_size) + 
                                                l;
                                input_index = (k * this->input_row_size * this->input_col_size) +
                                                ((j+m) * this->input_col_size) + 
                                                (i+l);
                                kernel_index = (n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
                                                (k * this->kernel_row_size * kernel_col_size) + 
                                                (j * this->kernel_col_size )+ 
                                                i;

                                intermidate = input[
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j+m) * this->input_col_size) + 
                                    (i+l)] *
                                kernels[
                                    (n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
                                    (k * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size )+ 
                                    i];
                            

                                output[
                                    (n * this->output_row_size * this->output_col_size) + 
                                    (m * this->output_col_size) + 
                                    l] += 

                                input[
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)] *
                                kernels[
                                    (n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
                                    (k * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size )+ 
                                    i];
                            }
                        }
                    }
                }
            }
        }
        break;

    case PADDING_SAME:
        // TODO
        break;

    }


}