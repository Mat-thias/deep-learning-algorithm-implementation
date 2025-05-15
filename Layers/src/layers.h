#ifndef LAYERS_H
#define LAYERS_H

#include "../../dlai.h"

// Layer type identifiers
#define FULLY_CONNECTED_LAYER       0x00
#define RELU_LAYER                  0x01

/**
 * @brief Abstract base class for all neural network layers.
 */
class Layer {
public:
    /**
     * @brief Performs the forward pass of the layer.
     * 
     * @param input Pointer to input data.
     * @param output Pointer to output data.
     */
    virtual void forward(float *input, float *output);
};

/**
 * @brief Fully connected (dense) layer implementation.
 * 
 * This layer performs a matrix-vector multiplication between the input
 * and the weight matrix, followed by bias addition.
 */
class FullyConnected : public Layer {
private:
    uint32_t input_size;     ///< Number of input nodes (features)
    uint32_t output_size;    ///< Number of output nodes
    float *weights;          ///< Pointer to weight matrix (size: output_size × input_size)
    float *bias;             ///< Pointer to bias vector (size: output_size)

public:
    /**
     * @brief Constructor for the FullyConnected layer.
     * 
     * @param input_size_ Number of input nodes.
     * @param output_size_ Number of output nodes.
     * @param weights_ Pointer to the weight matrix.
     * @param bias_ Pointer to the bias vector.
     */
    FullyConnected(uint32_t input_size, uint32_t output_size, float *weights, float *bias);

    /**
     * @brief Performs the forward pass for the fully connected layer.
     * 
     * @param input Pointer to input data.
     * @param output Pointer to output data.
     */
    void forward(float *input, float *output);
}; 

/**
 * @brief ReLU (Rectified Linear Unit) activation layer implementation.
 * 
 * Applies the element-wise activation function: max(0, x)
 */
class Relu : public Layer {
private:
    uint32_t input_dim;         ///< Number of input dimensions (e.g., 1 for 1D, 2 for 2D)
    uint32_t *input_shape;      ///< Pointer to shape array of the input tensor

public:
    /**
     * @brief Constructor for the ReLU layer.
     * 
     * @param input_dim Number of dimensions of the input.
     * @param input_shape Pointer to an array representing the shape of the input.
     */
    Relu(uint32_t input_dim, uint32_t *input_shape);

    /**
     * @brief Performs the forward pass for the ReLU activation.
     * 
     * @param input Pointer to input data.
     * @param output Pointer to output data.
     */
    void forward(float *input, float *output);
};

#endif // LAYERS_H




/**
 * @brief 2D Convolutional layer implementation.
 * 
 * Applies a set of convolutional kernels to 2D input data to produce 2D output feature maps.
 */
class Convolutional2DLayer : public Layer {
    private:
        uint32_t input_row_size;     ///< Number of rows in the input matrix
        uint32_t input_col_size;     ///< Number of columns in the input matrix
        uint32_t output_row_size;    ///< Number of rows in the output matrix
        uint32_t output_col_size;    ///< Number of columns in the output matrix
        float *kernels;              ///< Pointer to the filter/kernel weights
        float *bias;                 ///< Pointer to the bias values

    public:
        /**
         * @brief Constructor for the Convolutional2DLayer.
         * 
         * @param input_row_size_ Number of rows in the input.
         * @param input_col_size_ Number of columns in the input.
         * @param output_row_size_ Number of rows in the output.
         * @param output_col_size_ Number of columns in the output.
         * @param kernels_ Pointer to the kernel/filter weights.
         * @param bias_ Pointer to the bias values.
         */
        Convolutional2DLayer(uint32_t input_row_size_, uint32_t input_col_size_, 
                                uint32_t output_row_size_, uint32_t output_col_size_, 
                                float *kernels_, float *bias_);

        /**
         * @brief Performs the forward pass for the 2D convolutional layer.
         * 
         * @param input Pointer to input data.
         * @param output Pointer to output data.
         */
        void forward(float *input, float *output);
};    
