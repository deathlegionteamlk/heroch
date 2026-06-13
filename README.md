# Heroch: High-Performance Deep Learning and Scientific Computing Library

[![PyPI version](https://badge.fury.io/py/heroch.svg)](https://badge.fury.io/py/heroch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Heroch** is a powerful, flexible, and intuitive Python library designed for **machine learning**, **deep learning**, and **scientific computing**. It provides a high-performance tensor computation platform with seamless **GPU acceleration** (via CUDA/CuPy) and a dynamic **automatic differentiation** (autograd) engine.

## Why Heroch?

Heroch is built to feel like a native Python library, integrating perfectly with the SciPy ecosystem. Whether you are conducting academic research or building production-level neural networks, Heroch offers the flexibility to experiment and the speed to scale.

### Key Features

*   **⚡ High-Performance Tensor Computations**: Native NumPy interface with optional GPU acceleration for massive speedups.
*   **🧠 Dynamic Autograd Engine**: Flexible computational graphs that allow you to change network structures on the fly.
*   **🛠️ Complete Deep Learning Platform**: Built-in modules for linear layers, common activations (ReLU, Sigmoid), and loss functions (MSELoss).
*   **🚀 Optimized Training**: Robust optimizers including Stochastic Gradient Descent (SGD) with momentum.
*   **🐍 Pythonic API**: Intuitive design that is easy to debug and executes code as you write it.

## Installation

Install Heroch easily via pip:

```bash
pip install heroch
```

*Note: For GPU support, ensure you have an NVIDIA GPU and the appropriate CUDA drivers installed.*

## Quick Start Example

Build and train a simple neural network in minutes:

```python
import heroch
import heroch.nn as nn
import heroch.optim as optim
from heroch import Tensor

# Define your flexible model structure
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model, loss, and optimizer
model = MyNeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training data
inputs = Tensor([[1.0, 2.0], [3.0, 4.0]])
targets = Tensor([[5.0], [11.0]])

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
```

## Advanced Usage: GPU Acceleration

Heroch makes it easy to move your computations to the GPU:

```python
# Move model parameters and data to GPU
x_gpu = Tensor([1.0, 2.0, 3.0]).to_gpu()
```

## Contributing

We welcome contributions from the community! Check out our [GitHub repository](https://github.com/death-legion-team/heroch) to get involved.

---
Developed with 🔥 by **Death Legion Team**.
