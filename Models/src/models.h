#ifndef MODELS_H
#define MODELS_H

#include "../../dlai.h"
#include "../../Layers/src/layers.h"

#include <stdio.h>

/**
 * @brief A class that represents a sequential model consisting of layers.
 * 
 * This class allows running inference (prediction) on input data by sequentially 
 * passing it through a predefined list of neural network layers. The user must
 * set the `input` pointer before calling `predict()`, and the result will be 
 * available in the `output` pointer after prediction.
 */
class Sequential {

private:
    Layer **graph;                  ///< Pointer to an array of Layer pointers forming the model graph.
    uint32_t no_layers;             ///< Number of layers in the model/model graph.
    uint32_t worksheet_arena_size;  ///< Size of the worksheet buffer for intermediate outputs.
    float *worksheet;               ///< Shared buffer used for intermediate layer outputs.

public:
    float *input;                   ///< Pointer to input data buffer.
    float *output;                  ///< Pointer to final output buffer.

    /**
     * @brief Constructor to initialize the model with an external graph and worksheet.
     * 
     * This allows for externally constructed layer graphs and shared memory usage.
     * 
     * @param model_arr Pointer to the serialized model byte array.
     * @param model_len Total length of the model array in bytes.
     * @param graph     Pre-allocated array of Layer pointers representing the model.
     * @param no_layers Number of layers in the model/model graph.
     * @param worksheet Pre-allocated worksheet memory for intermediate outputs.
     * @param worksheet_arena_size Size (in floats) of the worksheet buffer.
     */
    Sequential(uint8_t *model_arr, uint32_t model_len, Layer **graph, uint32_t no_layers, float *worksheet, uint32_t worksheet_arena_size);

    /**
     * @brief Performs forward propagation through the model to generate predictions.
     * 
     * The `input` pointer must point to valid input data before this function is called.
     * The model will sequentially process the data through all layers and store
     * the final result in the memory pointed to by `output`.
     */
    void predict(void);
};

#endif // MODELS_H
