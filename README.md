# DLAI (Deep Learning Algorithm Implementation)

DLAI is a streamlined repository focused on implementing core deep learning algorithms from scratch in C/C++. It is designed primarily for educational purposes and optimized for deployment on embedded systems and microcontrollers, supporting TinyML/EmbeddedAI applications. This repository serves as a foundational reference for understanding core deep learning principles and exploring TinyML applications.

---

## 🧠 Current State

- The project currently supports **conversion of sequential neural networks from the PyTorch library only**.
- It uses `torch.save()` to save the full model, and loads it using:
```python
  torch.load(pth_file_path, weight_only=False)
```
### ✅ Layers Currently Supported

0. Linear (Fully Connected) Layer  
1. ReLU Layer  

> Additional layers and features are actively under development.

---

## 🚀 Features

- **Algorithm Implementations**: Basic neural network layers implemented from scratch.
- **Embedded Systems Focus**: Optimized for TinyML and embedded system deployment.
- **Educational Resource**: A practical reference for learning deep learning fundamentals and embedded deployment.

---

## 🛠️ Getting Started

### 1. Clone the Repository
```
git clone https://github.com/Mat-thias/dlai.git
```
### 2. Navigate to the Project Directory
```
cd dlai
```
### 3. Convert a PyTorch Sequential Model to a C Header File

Run the script located in `dlai/Models`:
```
python3 dlai/Models/convert_sequential_model_to_c.py <model_file> <model_name> <input_shape> [output_dir]
```
> Note: It is reocommended to create a virtual environment and install the packages in `dlai/Models/venv_requirement.txt`
#### Example
```
python3 dlai/Models/convert_sequential_model_to_c.py dlai/examples/sine_model/sine_model.pth sine_model "(1,1)" dlai/examples/sine_model/
```
---

## ⚙️ Example Usage

```cpp
Layer *graph[LAYER_LEN];
float workspace[MAX_WORKSPACE_SIZE];

Sequential model(sine_model, sine_model_len, graph, LAYER_LEN, workspace, MAX_WORKSPACE_SIZE);

float *input = model.input;
float *output = model.output;

*input = (float)i * 2 * 3.141 / 360;
model.predict();
Serial.println(*output);
```
---

## 📂 Explore the Implementations

* Contains **C/C++ source code** for supported layers.
* Each implemented component includes inline documentation.

---

## 🤝 Contributing

We welcome contributions to expand supported layers and improve the tool!

1. **Fork the Repository** on GitHub.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/your-username/dlai.git
   ```

3. **Create a Feature Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**.

5. **Commit Your Changes**:

   ```bash
   git commit -m "Add your commit message"
   ```

6. **Push to Your Fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** to the main repository.

