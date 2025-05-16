#include "models.h"

/**
 * @brief Constructs a Sequential model from a serialized model array.
 * 
 * Parses the provided byte array to populate the model’s computation graph
 * with layer instances (e.g., FullyConnected, ReLU). The layer graph and 
 * workspace must be pre-allocated by the caller.
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

    // Read and validate number of layers
    if (layer_len < *(uint32_t *)model_arr) {
        return;
    }
    model_arr += sizeof(uint32_t);
    this->graph = graph;
    this->layer_len = layer_len;

    // Read and validate required workspace size
    if (workspace_size < (*(uint32_t *)model_arr * 2)) {
        return;
    }
    model_arr += sizeof(uint32_t);
    this->workspace = workspace;
    this->workspace_size = workspace_size;

    int i = 0;

    // Parse and instantiate layers from serialized data
    while (model_arr < start + model_len) {
        uint32_t layer_type = *model_arr;
        model_arr += sizeof(uint32_t);

        switch (layer_type) {
            
            case FULLY_CONNECTED_LAYER: {
                uint32_t input_size = *(uint32_t*)model_arr;
                model_arr += sizeof(uint32_t);

                uint32_t output_size = *(uint32_t*)model_arr;
                model_arr += sizeof(uint32_t);

                float *weight = (float *)model_arr;
                model_arr += sizeof(float) * input_size * output_size;

                float *bias = (float *)model_arr;
                model_arr += sizeof(float) * output_size;

                this->graph[i] = new FullyConnected(input_size, output_size, weight, bias);
                break;
            }

            case RELU_LAYER: {
                uint32_t input_dim = *(uint32_t*)model_arr;
                model_arr += sizeof(uint32_t);

                uint32_t *input_shape = (uint32_t *)model_arr;
                model_arr += sizeof(uint32_t) * input_dim;

                this->graph[i] = new Relu(input_dim, input_shape);
                break;
            }

            default:
                // Unknown layer type: skip or handle error if needed
                break;
        }

        i++;
    }
    printf("%d", i);
    input = this->workspace;
    if (i%2 == DLAI_EVEN) {output = this->workspace;}
    else {output = workspace + (this->workspace_size / 2);}
}

/**
 * @brief Runs inference using the model, processing input through all layers.
 * 
 * Utilizes a double-buffering strategy within a single workspace array to 
 * alternate between read and write buffers, minimizing memory allocation overhead.
 * 
 * @param input Pointer to the input data array.
 * @param output Pointer to the output data array to be filled.
 */
void Sequential::predict(void) {
    for (int i = 0; i < this->layer_len; i++) {
        int input_workspace_offset;
        int output_workspace_offset;

        // Alternate between two halves of the workspace buffer
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

        // if (i == 0) {
        //     // First layer takes input from the original input array
        //     this->graph[0]->forward(input, this->workspace + output_workspace_offset);
        // }
        // else if (i == layer_len - 1) {
        //     // Final layer writes output directly to the output array
        //     this->graph[i]->forward(this->workspace + input_workspace_offset, output);
        // }
        // else {
            // Intermediate layers read from and write to workspace buffer
            this->graph[i]->forward(this->workspace + input_workspace_offset, this->workspace + output_workspace_offset);
        // }
    }
}
