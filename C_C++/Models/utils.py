import struct
import numpy as np
import tensorflow as tf

# Layer type identifiers
FULLY_CONNECTED_LAYER_TYPE = 0
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
# @param model Keras model (not yet used here).
# @return Integer size of the workspace arena (default: 1024).
# ------------------------------------------------------------------------------
def get_max_workspace_arena(model):
    # To be implemented: dynamic calculation based on model
    return 32

# ------------------------------------------------------------------------------
# @brief Converts a Keras Sequential model into a byte array formatted
#        as a C header file for embedded usage.
#
# @param model A Keras Sequential model object.
# @param model_name Name to use in the C header file.
# @return A string containing the C array definition.
# ------------------------------------------------------------------------------
def convert_sequential_model_to_c_format(model, model_name):
    c_array = f"#ifndef {model_name.upper()}_H\n#define {model_name.upper()}_H\n\nunsigned char {model_name.lower()}[] = {{\n"
  
    # Determine number of layers, accounting for activations
    if hasattr(model.layers[-1], 'activation') and model.layers[-1].activation.__name__ == "linear":
        model_len = len(model.layers) * 2 - 1
    else:
        model_len = len(model.layers) * 2

    c_array += "    // Number of Layers\n"
    c_array += "    " + ", ".join([f"0x{b:02X}" for b in int32_to_bytes(model_len)]) + ",\n"

    # Add max workspace arena
    max_workspace_arena = get_max_workspace_arena(model)
    c_array += "    // Max workspace arena\n"
    c_array += "    " + ", ".join([f"0x{b:02X}" for b in int32_to_bytes(max_workspace_arena)]) + ",\n"

    # Encode each layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights, bias = layer.get_weights()
            input_size, output_size = weights.shape

            model_type = FULLY_CONNECTED_LAYER_TYPE

            c_array += "\n    // Layer Type: Fully Connected Layer\n"
            c_array += "    " + ", ".join([f"0x{b:02X}" for b in int32_to_bytes(model_type)]) + ",\n"

            c_array += "    // Input size and Output size\n"
            c_array += "    " + ", ".join([f"0x{b:02X}" for b in (
                int32_to_bytes(input_size) + int32_to_bytes(output_size)
            )]) + ",\n"

            section = max(len(weights.flatten()) // 4, 1)

            c_array += "    // Weights\n"
            for line in np.array_split(weights.flatten(), section):
                c_array += "    " + ", ".join(
                    [f"0x{b:02X}" for val in line for b in float32_to_bytes(val)]
                ) + ",\n"

            section = max(len(bias.flatten()) // 4, 1)

            c_array += "    // Biases\n"
            for line in np.array_split(bias.flatten(), section):
                c_array += "    " + ", ".join(
                    [f"0x{b:02X}" for val in line for b in float32_to_bytes(val)]
                ) + ",\n"

        if hasattr(layer, 'activation') and layer.activation.__name__ == "relu":
            model_type = RELU_LAYER
            input_dim = 1
            input_shape = output_size

            c_array += "\n    // Layer Type: ReLU Activation\n"
            c_array += "    " + ", ".join(
                [f"0x{b:02X}" for b in int32_to_bytes(model_type)]
            ) + ",\n"

            c_array += "    // Input dimensions and shape\n"
            c_array += "    " + ", ".join(
                [f"0x{b:02X}" for b in (
                    int32_to_bytes(input_dim) + int32_to_bytes(input_shape)
                )]
            ) + ",\n"

    c_array += f"}};\n\nunsigned int {model_name.lower()}_len = sizeof({model_name.lower()});\n\n#endif // {model_name.upper()}_H\n"

    return c_array

# ------------------------------------------------------------------------------
# Load the model and print its C-format representation
# ------------------------------------------------------------------------------
model_name = "sine_model"
# print(convert_sequential_model_to_c_format(tf.keras.models.load_model("sine_model.keras"), model_name))

with open(f"../examples/sine_model/{model_name}.h", "w") as f:
    f.write(convert_sequential_model_to_c_format(tf.keras.models.load_model("../examples/sine_model/sine_model.keras"), model_name))

