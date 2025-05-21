import sys
import ast

import numpy as np
import torch 
from torch import nn

import struct
from os import path

# Layer type identifiers (used to differentiate layer types during serialization)
LINEAR_LAYER_TYPE = 0
RELU_LAYER = 1

# ------------------------------------------------------------------------------
# @brief Converts a float32 value to a list of 4 bytes in little-endian format.
# @param val Float value to convert.
# @return List of 4 bytes representing the float.
# ------------------------------------------------------------------------------
def float32_to_bytes(val):
    return list(struct.pack("<f", val))  # Little endian float32

# ------------------------------------------------------------------------------
# @brief Converts an int32 value to a list of 4 bytes in little-endian format.
# @param val Integer value to convert.
# @return List of 4 bytes representing the integer.
# ------------------------------------------------------------------------------
def int32_to_bytes(val):
    return list(struct.pack("<i", val))  # Little endian int32

# ------------------------------------------------------------------------------
# @brief Returns the maximum workspace arena size.
# @param seqential_model Torch Sequential model.
# @return Integer representing the maximum number of floats needed in workspace.
# ------------------------------------------------------------------------------
def get_max_workspace_arena(seqential_model, input_shpae):
    max_output = 0

    # Create random input tensor based on model's expected input shape
    x = torch.randn(input_shape)
    
    for layer in seqential_model:
        x = layer(x)
        max_output = max(max_output, x.numel())  # Track max number of elements
    return max_output

# ------------------------------------------------------------------------------
# @brief Converts a PyTorch Sequential model into a C header file.
# @param seqential_model PyTorch model object.
# @param model_name Name used for C array and header file.
# @param dir Directory to write the header file into.
# ------------------------------------------------------------------------------
def convert_sequential_model_to_c(seqential_model, model_name, input_shape, dir="."):
    # Start building the C array
    c_array = f"#ifndef {model_name.upper()}_H\n#define {model_name.upper()}_H\n\nunsigned char {model_name.lower()}[] = {{\n"

    model_len = len(seqential_model)

    # Add number of layers
    c_array += "    // Number of Layers\n"
    c_array += "    " + ", ".join([f"0x{b:02X}" for b in int32_to_bytes(model_len)]) + ",\n"

    # Add workspace arena size
    max_workspace_arena = get_max_workspace_arena(seqential_model, input_shape)
    c_array += "    // Max workspace arena\n"
    c_array += "    " + ", ".join([f"0x{b:02X}" for b in int32_to_bytes(max_workspace_arena)]) + ",\n"

    # Serialize each layer
    for i, layer in enumerate(seqential_model):
        if isinstance(layer, nn.Linear):
            # Extract weights and biases
            weight = layer.weight.cpu().detach().numpy()
            bias = layer.bias.cpu().detach().numpy()

            output_size, input_size = weight.shape
            model_type = LINEAR_LAYER_TYPE

            # Serialize linear layer metadata
            c_array += "\n    // Layer Type: Linear Layer\n"
            c_array += "    " + ", ".join([f"0x{b:02X}" for b in int32_to_bytes(model_type)]) + ",\n"

            c_array += "    // Output size and Input size\n"
            c_array += "    " + ", ".join([f"0x{b:02X}" for b in (
                int32_to_bytes(output_size) + int32_to_bytes(input_size)
            )]) + ",\n"

            # Serialize weights
            section = max(len(weight.flatten()) // 4, 1)
            c_array += "    // Weight\n"
            for line in np.array_split(weight.flatten(), section):
                c_array += "    " + ", ".join(
                    [f"0x{b:02X}" for val in line for b in float32_to_bytes(val)]
                ) + ",\n"

            # Serialize biases
            section = max(len(bias.flatten()) // 4, 1)
            c_array += "    // Bias\n"
            for line in np.array_split(bias.flatten(), section):
                c_array += "    " + ", ".join(
                    [f"0x{b:02X}" for val in line for b in float32_to_bytes(val)]
                ) + ",\n"

        elif isinstance(layer, nn.ReLU):
            # Get input shape from previous layer's output
            model_type = RELU_LAYER
            input_shape = seqential_model[i-1].bias.cpu().detach().numpy().shape
            input_dim = len(input_shape)
            
            input_size = 1
            for dim in range(input_dim):
                input_size *= input_shape[dim]

            # Serialize ReLU layer metadata
            c_array += "\n    // Layer Type: ReLU Activation\n"
            c_array += "    " + ", ".join(
                [f"0x{b:02X}" for b in int32_to_bytes(model_type)]
            ) + ",\n"

            c_array += "    // Input dimensions and shape\n"
            c_array += "    " + ", ".join(
                [f"0x{b:02X}" for b in (
                    int32_to_bytes(input_dim) + int32_to_bytes(input_size)
                )]
            ) + ",\n"

    # Finalize the header file
    c_array += f"}};\n\nunsigned int {model_name.lower()}_len = sizeof({model_name.lower()});\n\n#endif // {model_name.upper()}_H\n"

    # Write to file
    with open(path.join(dir, f"{model_name}.h"), "w") as file:
        file.write(c_array)

# ------------------------------------------------------------------------------
# @brief Entry point when the script is run as a standalone module.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Invalid arguments passed.\nFormat python3 convert_sequential_model_to_c.py seqential_model_file model_name input_shape [c_file_dir]")
        sys.exit()

    seqential_model_file = sys.argv[1]
    model_name = sys.argv[2]
    input_shape_str = sys.argv[3]
    c_file_dir = sys.argv[4] if len(sys.argv) == 5 else "."

    seqential_model = torch.load(seqential_model_file, weights_only=False).to("cpu")
    input_shape = tuple(ast.literal_eval(input_shape_str))

    convert_sequential_model_to_c(seqential_model, model_name, input_shape, c_file_dir)
