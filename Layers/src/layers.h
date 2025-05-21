#ifndef LAYERS_H
#define LAYERS_H

#include "../../dlai.h"

// Layer type identifiers for differentiating layer classes
#define FULLY_CONNECTED_LAYER       0x00
#define RELU_LAYER                  0x01

/**
 * @brief Abstract base class for all neural network layers.
 * 
 * Provides a virtual interface for the forward method to be implemented
 * by all derived layer types.
 */
class Layer {
public:
    /**
     * @brief Virtual function for the forward pass of a layer.
     * 
     * This should be overridden by derived layer classes.
     * 
     * @param input Pointer to the input data array.
     * @param output Pointer to the output data array.
     */
    virtual void forward(float *input, float *output);
};

/**
 * @brief Fully connected (dense) layer.
 * 
 * Computes output = input × weights^T + bias.
 */
class Linear : public Layer {
private:
    uint32_t input_size;      ///< Number of input features
    uint32_t output_size;     ///< Number of output neurons
    float *weights;           ///< Weight matrix (output_size × input_size)
    float *bias;              ///< Bias vector (size: output_size)

public:
    /**
     * @brief Constructor for Linear (fully connected) layer.
     * 
     * @param output_size Number of output neurons.
     * @param input_size Number of input features.
     * @param weights Pointer to weight matrix.
     * @param bias Pointer to bias vector.
     */
    Linear(uint32_t output_size, uint32_t input_size, float *weights, float *bias);

    /**
     * @brief Forward pass: computes the dense layer output.
     * 
     * @param input Pointer to input array.
     * @param output Pointer to output array.
     */
    void forward(float *input, float *output);
};

/**
 * @brief ReLU (Rectified Linear Unit) activation layer.
 * 
 * Applies an element-wise max(0, x) operation.
 */
class ReLU : public Layer {
private:
    uint32_t input_dim;         ///< Number of dimensions of the input tensor
    uint32_t *input_shape;      ///< Array indicating shape of input tensor

public:
    /**
     * @brief Constructor for ReLU layer.
     * 
     * @param input_dim Dimensionality of the input.
     * @param input_shape Pointer to shape array.
     */
    ReLU(uint32_t input_dim, uint32_t *input_shape);

    /**
     * @brief Applies ReLU activation on input.
     * 
     * @param input Pointer to input array.
     * @param output Pointer to output array.
     */
    void forward(float *input, float *output);
};

/**
 * @brief 2D Convolutional layer.
 * 
 * Applies a 2D convolution using multiple kernels across the input.
 */
class Convolutional2DLayer : public Layer {
private:
    uint32_t input_channel_size;     ///< Number of input channels
    uint32_t input_row_size;         ///< Height of input feature map
    uint32_t input_col_size;         ///< Width of input feature map

    uint32_t output_channel_size;    ///< Number of output channels
    uint32_t output_row_size;        ///< Height of output feature map
    uint32_t output_col_size;        ///< Width of output feature map

    uint32_t kernel_row_size;        ///< Height of kernel
    uint32_t kernel_col_size;        ///< Width of kernel

    uint32_t stride_row;             ///< Stride in vertical direction
    uint32_t stride_col;             ///< Stride in horizontal direction
    uint32_t padding;                ///< Padding size around input

    float *kernels;                  ///< Pointer to convolution kernels
    float *bias;                     ///< Pointer to bias array

public:
    /**
     * @brief Constructor for 2D convolutional layer.
     * 
     * @param input_channel_size Number of input channels.
     * @param input_row_size Height of input.
     * @param input_col_size Width of input.
     * @param output_channel_size Number of filters/output channels.
     * @param kernel_row_size Height of kernel.
     * @param kernel_col_size Width of kernel.
     * @param stride_row Vertical stride.
     * @param stride_col Horizontal stride.
     * @param padding Padding size.
     * @param kernels Pointer to kernel weights.
     * @param bias Pointer to bias values.
     */
    Convolutional2DLayer(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size, 
                         uint32_t output_channel_size, int32_t kernel_row_size, uint32_t kernel_col_size,
                         uint32_t stride_row, uint32_t stride_col, uint32_t padding,
                         float *kernels, float *bias);

    /**
     * @brief Applies the convolutional operation on input.
     * 
     * @param input Pointer to input array.
     * @param output Pointer to output array.
     */
    void forward(float *input, float *output);
};

/**
 * @brief 2D Max Pooling layer.
 * 
 * Reduces spatial dimensions by selecting max values in each window.
 */
class MaxPooling2DLayer : public Layer {
private:
    uint32_t input_channel_size; ///< Number of input channels
    uint32_t input_row_size;     ///< Height of input feature map
    uint32_t input_col_size;     ///< Width of input feature map

    uint32_t output_row_size;    ///< Height of output feature map
    uint32_t output_col_size;    ///< Width of output feature map

    uint32_t pool_row;           ///< Height of pooling window
    uint32_t pool_col;           ///< Width of pooling window
    uint32_t stride_row;         ///< Stride in vertical direction
    uint32_t stride_col;         ///< Stride in horizontal direction
    uint32_t padding;            ///< Padding size around input

public:
    /**
     * @brief Constructor for 2D max pooling layer.
     * 
     * @param input_channel_size Number of input channels.
     * @param input_row_size Input height.
     * @param input_col_size Input width.
     * @param pool_row Pooling window height.
     * @param pool_col Pooling window width.
     * @param stride_row Vertical stride.
     * @param stride_col Horizontal stride.
     * @param padding Padding size.
     */
    MaxPooling2DLayer(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
                      uint32_t pool_row, uint32_t pool_col,
                      uint32_t stride_row, uint32_t stride_col, uint32_t padding);

    /**
     * @brief Applies max pooling to the input data.
     * 
     * @param input Pointer to input array.
     * @param output Pointer to output array.
     */
    void forward(float *input, float *output);
};

#endif // LAYERS_H
