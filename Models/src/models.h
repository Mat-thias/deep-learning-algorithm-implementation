#ifndef MODELS_H
#define MODELS_H

#include "../../dlai.h"
#include "../../Layers/src/layers.h"

#include <stdio.h>

/**
 * @brief A class that represents a sequential model consisting of layers.
 * 
 * This class allows running inference (prediction) on input data by sequentially 
 * passing it through a predefined list of neural network layers.
 */
class Sequential {

private:
    Layer **graph;              ///< Pointer to an array of Layer pointers forming the model graph.
    uint32_t layer_len;         ///< Total number of layers in the model.
    uint32_t workspace_size;    ///< Size of the workspace buffer for intermediate outputs.
    float *workspace;           ///< Single buffer used for intermediate computations.



public:

    float *input;
    float *output;
    /**
     * @brief Constructor to initialize the model with an external graph and workspace.
     * 
     * This allows for externally constructed layer graphs and shared memory usage.
     * 
     * @param model_arr Pointer to the serialized model byte array.
     * @param model_len Total length of the model array in bytes.
     * @param graph Pre-allocated array of Layer pointers (model graph).
     * @param layer_len Number of layers in the graph.
     * @param workspace Pre-allocated workspace memory for intermediate outputs.
     * @param workspace_size Size (in floats) of the workspace buffer.
     */
    Sequential(uint8_t *model_arr, uint32_t model_len, Layer **graph, uint32_t layer_len, float *workspace, uint32_t workspace_size);

    /**
     * @brief Performs forward propagation through the model to generate predictions.
     * 
     * The input is passed through each layer sequentially, and the final output is
     * stored in the output buffer.
     * 
     * @param input Pointer to the input data array.
     * @param output Pointer to the output data array to be filled.
     */
    void predict(void);
};

#endif // MODELS_H
