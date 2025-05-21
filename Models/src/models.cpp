#include "models.h"
#include <stdio.h>

/**
 * @brief Constructs a Sequential model from a serialized model array.
 * 
 * Parses the provided byte array to populate the model’s computation graph
 * with instantiated Layer objects (e.g., Linear, ReLU). The caller must pre-allocate:
 * - An array of Layer pointers to hold the model layers.
 * - A workspace buffer for intermediate computation results.
 * 
 * The constructor also initializes the `input` and `output` pointers based on
 * double-buffering logic.
 * 
 * @param model_arr Pointer to the serialized model byte array.
 * @param model_len Total length of the model array in bytes.
 * @param graph Pre-allocated array for storing layer pointers.
 * @param layer_len Length of the graph array (number of layers it can hold).
 * @param workspace Pre-allocated workspace memory for intermediate outputs.
 * @param workspace_size Size of the workspace memory (in floats).
 */
Sequential::Sequential(uint8_t *model_arr, uint32_t model_len, Layer **graph, uint32_t layer_len, float *workspace, uint32_t workspace_size) {
    unsigned char *start = model_arr;

    // Read and validate number of layers from serialized header
    if (layer_len < *(uint32_t *)model_arr) {
        return;
    }
    model_arr += sizeof(uint32_t);
    this->graph = graph;
    this->layer_len = layer_len;

    // Read and validate workspace size requirement
    if (workspace_size < (*(uint32_t *)model_arr * 2)) {
        return;
    }
    model_arr += sizeof(uint32_t);
    this->workspace = workspace;
    this->workspace_size = workspace_size;

    int i = 0;

    // Parse layers from serialized model array
    while (model_arr < start + model_len) {
        uint32_t layer_type = *model_arr;
        model_arr += sizeof(uint32_t);

        switch (layer_type) {
            case FULLY_CONNECTED_LAYER: {
                // Deserialize fully connected layer parameters
                uint32_t output_size = *(uint32_t*)model_arr;
                model_arr += sizeof(uint32_t);

                uint32_t input_size = *(uint32_t*)model_arr;
                model_arr += sizeof(uint32_t);

                float *weight = (float *)model_arr;
                model_arr += sizeof(float) * output_size * input_size;

                float *bias = (float *)model_arr;
                model_arr += sizeof(float) * output_size;

                this->graph[i] = new Linear(output_size, input_size, weight, bias);
                break;
            }

            case RELU_LAYER: {
                // Deserialize ReLU layer parameters
                uint32_t input_dim = *(uint32_t*)model_arr;
                model_arr += sizeof(uint32_t);

                uint32_t *input_shape = (uint32_t *)model_arr;
                model_arr += sizeof(uint32_t) * input_dim;

                this->graph[i] = new ReLU(input_dim, input_shape);
                break;
            }

            default:
                // Unknown layer type: skip or handle error
                break;
        }

        i++;
    }

    // Set model input/output buffers based on double-buffering strategy
    input = this->workspace;
    if (i % 2 == DLAI_EVEN) {
        output = this->workspace;
    } else {
        output = workspace + (this->workspace_size / 2);
    }
}

/**
 * @brief Runs inference by passing input through each layer sequentially.
 * 
 * Uses a double-buffering approach in a single contiguous workspace to alternate
 * between input and output for each layer, avoiding repeated memory allocations.
 * 
 * The result of the final layer will be written to the memory pointed to by `output`.
 */
void Sequential::predict(void) {
    for (int i = 0; i < this->layer_len; i++) {
        int input_workspace_offset;
        int output_workspace_offset;

        // Alternate buffer halves between layers for intermediate results
        switch (i % 2) {
            case DLAI_EVEN:
                input_workspace_offset = 0;
                output_workspace_offset = this->workspace_size / 2;
                break;
            default:
                input_workspace_offset = this->workspace_size / 2;
                output_workspace_offset = 0;
                break;
        }

        // Apply forward pass of the i-th layer
        this->graph[i]->forward(
            this->workspace + input_workspace_offset, 
            this->workspace + output_workspace_offset
        );
    }
}
