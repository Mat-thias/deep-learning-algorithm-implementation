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