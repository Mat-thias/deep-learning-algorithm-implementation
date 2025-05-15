#include <iostream>

// Include the pre-trained sine model weights and configuration
#include "sine_model.h"

// Include the custom layers and models definitions
#include "../../Layers/src/layers.h"
#include "../../Models/src/models.h"

// Define the number of layers in the network
#define LAYER_LEN 5

// Define the maximum workspace size for intermediate computation
#define MAX_WORKSPACE_SIZE 1024 * 2

// Declare a static array of layer pointers to represent the model graph
Layer *graph[LAYER_LEN];

// Allocate workspace memory used for internal computations (activations, temp buffers)
float workspace[MAX_WORKSPACE_SIZE];

// Initialize a sequential model with:
// - sine_model: pointer to model parameters (weights and biases)
// - sine_model_len: total number of parameters
// - graph: pointer to the layers array to be constructed internally
// - LAYER_LEN: number of layers in the model
// - workspace: preallocated memory for temporary computations
// - MAX_WORKSPACE_SIZE: size of the workspace
Sequential model(sine_model, sine_model_len, graph, LAYER_LEN, workspace, MAX_WORKSPACE_SIZE);

int main() {
    std::cout << "Hello World\n";

    // Input to the network: a single float representing an angle in radians
    float x = 1;
    float *input = &x;

    // Output of the network: the predicted sine value
    float y = 0;
    float *output = &y;
    
    // Loop over 360 degrees and predict the sine value using the model
    for (int i = 0; i < 360; i++) {
        // Convert degrees to radians
        *input = (float)i * 2 * 3.141 / 360;

        // Predict using the model
        model.predict(input, output);

        // Print input angle (in degrees) and the predicted sine value
        std::cout << "input " << i << " output " << *output << "\n"; 
    }
}
