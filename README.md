<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=5,10,24&height=220&section=header&text=Heroch&fontSize=85&fontColor=ffffff&fontAlignY=38&desc=Deep%20learning%20%26%20scientific%20computing%20in%20Python&descAlignY=60&descSize=20&animation=fadeIn" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2800&pause=900&color=F59E0B&center=true&vCenter=true&multiline=true&width=680&height=80&lines=NumPy+interface.+CUDA+acceleration.+Autograd.;Build+nets.+Train+fast.+Run+on+GPU.;Python-native.+SciPy-compatible." alt="Typing animation"/>

<br/><br/>

[![PyPI](https://img.shields.io/pypi/v/heroch?style=for-the-badge&logo=python&logoColor=white&color=f59e0b)](https://badge.fury.io/py/heroch)
[![License: MIT](https://img.shields.io/badge/License-MIT-6366f1?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-Optional-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Built by](https://img.shields.io/badge/💀-Death%20Legion%20Team-1a1a1a?style=for-the-badge)](https://github.com/death-legion-team)

</div>

---

## 🧠 What is Heroch?

Most deep learning libraries make you choose between two things: a clean Python interface, or actual performance. Heroch doesn't make you choose.

It's a tensor computation library that looks and feels like NumPy — same indexing, same broadcasting rules, same ecosystem compatibility — but with an autograd engine underneath and optional CUDA acceleration via CuPy. You write your forward pass, call `.backward()`, and the gradients flow. No graph compilation step, no static mode, no config file. Just Python running as you'd expect it to.

The API mirrors PyTorch closely enough that if you've used one, you can pick up the other in an afternoon. The difference is Heroch is built by the Death Legion Team, stays lightweight, and fits cleanly into the SciPy ecosystem without pulling in the entire world as a dependency.

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="600"/>
</div>

---

## ✨ What's in the box

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="80"/>
</div>

<table>
<tr>
<td width="50%">

### ⚡ High-Performance Tensors
The `Tensor` class wraps NumPy on CPU and CuPy on GPU. You get NumPy's full interface — slicing, broadcasting, reshaping — with a `.to_gpu()` call to move any tensor onto CUDA hardware.

### 🧠 Dynamic Autograd
The computational graph builds itself as your code runs. There's no separate "define then run" step — the graph exists when the forward pass does, and `.backward()` walks it to compute gradients. This means you can use plain Python control flow: `if` statements, loops, dynamic shapes — all of it works.

### 🔗 Neural Network Modules
`heroch.nn` has the layers you need to start: `Linear`, `ReLU`, `Sigmoid`, `MSELoss`, `CrossEntropyLoss`. They work the same way as PyTorch modules — subclass `nn.Module`, define `__init__` and `forward`, done.

</td>
<td width="50%">

### 🚀 Optimizers
`heroch.optim` ships with `SGD` (with momentum support) and `Adam`. Call `optimizer.zero_grad()` before your forward pass, `loss.backward()` after, `optimizer.step()` to update weights.

### 🐍 Pythonic by design
Everything executes eagerly. No deferred execution, no session objects, no graph compilation. You can drop a `print()` anywhere in your forward pass and it works. Debugging feels like debugging regular Python.

### 🔬 SciPy compatible
Heroch tensors convert to and from NumPy arrays with zero copying on CPU. Drop it into an existing scientific Python workflow — matplotlib, pandas, sklearn — without friction.

</td>
</tr>
</table>

---

## 📦 Install

```bash
pip install heroch
```

GPU support requires an NVIDIA GPU with CUDA drivers installed. Install CuPy separately for your CUDA version:

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

No CUDA? No problem — Heroch runs fully on CPU out of the box.

---

## 🚀 Quickstart

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="80"/>
</div>

### Build and train a neural network

```python
import heroch
import heroch.nn as nn
import heroch.optim as optim
from heroch import Tensor

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model     = MyNeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

inputs  = Tensor([[1.0, 2.0], [3.0, 4.0]])
targets = Tensor([[5.0], [11.0]])

for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")
```

---

### GPU acceleration

Moving to the GPU is one call per tensor. No global device context to manage.

```python
from heroch import Tensor

# CPU tensor
x = Tensor([1.0, 2.0, 3.0])

# Move to GPU (requires CUDA + CuPy)
x_gpu = x.to_gpu()

# Move back to CPU for numpy interop
x_cpu = x_gpu.to_cpu()
```

For a full model, move inputs and targets before the training loop:

```python
inputs  = Tensor([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
targets = Tensor([[5.0], [11.0]]).to_gpu()
model   = MyNeuralNetwork().to_gpu()
```

---

### Autograd by itself

You don't need a full `nn.Module` to use autograd. Any tensor with `requires_grad=True` tracks operations.

```python
from heroch import Tensor

x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

# z = x² + y * 5
z = x ** 2 + y * 5

z.backward()

print(x.grad)  # dz/dx = 2x = 4.0
print(y.grad)  # dz/dy = 5.0
```

---

### Custom loss function

Because the graph is dynamic, you can write any loss function in plain Python:

```python
def huber_loss(predictions, targets, delta=1.0):
    diff = predictions - targets
    abs_diff = diff.abs()
    quadratic = (diff ** 2) * 0.5
    linear = abs_diff * delta - (delta ** 2) * 0.5
    # standard Python conditional works fine here
    return (quadratic * (abs_diff <= delta) + linear * (abs_diff > delta)).mean()

loss = huber_loss(predictions, targets)
loss.backward()
```

---

## 📋 API reference

### `heroch.Tensor`

| Method | Description |
|---|---|
| `Tensor(data, requires_grad=False)` | Create a tensor from a list or numpy array |
| `.to_gpu()` | Move to CUDA device (returns new tensor) |
| `.to_cpu()` | Move back to CPU |
| `.backward()` | Backpropagate from this tensor |
| `.numpy()` | Convert to a NumPy array |
| `.grad` | Accumulated gradient after `.backward()` |

### `heroch.nn`

| Module | Description |
|---|---|
| `nn.Module` | Base class for all models |
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.ReLU()` | Rectified linear unit |
| `nn.Sigmoid()` | Sigmoid activation |
| `nn.MSELoss()` | Mean squared error |
| `nn.CrossEntropyLoss()` | Cross-entropy for classification |

### `heroch.optim`

| Optimizer | Description |
|---|---|
| `optim.SGD(params, lr, momentum=0)` | Stochastic gradient descent |
| `optim.Adam(params, lr=1e-3)` | Adam optimizer |
| `.zero_grad()` | Clear accumulated gradients |
| `.step()` | Apply gradient update |

---

## 🏗️ How autograd works

```
Forward pass
   │
   ▼
Operations build a computation graph
   (each node stores its inputs and the op used)
   │
   ▼
loss.backward()
   │
   ▼
Chain rule applied backwards through the graph
   │
   ▼
.grad populated on every tensor with requires_grad=True
```

The graph exists only for the duration of one forward pass. After `.backward()`, it's released. This keeps memory usage predictable and means you never have to think about "clearing" a stale graph.

---

## 🤝 Contributing

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="400"/>
</div>

PRs and issues welcome. For new features, open an issue first so we can agree on direction before you write the code.

```bash
git clone https://github.com/death-legion-team/heroch.git
cd heroch
pip install -e ".[dev]"
pytest
```

---

## 🛡️ License

MIT © [Death Legion Team](https://github.com/death-legion-team)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=5,10,24&height=100&section=footer&animation=fadeIn" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&duration=4000&pause=1000&color=F59E0B&center=true&vCenter=true&width=540&lines=NumPy+interface.+Autograd.+CUDA.;Write+Python.+Get+performance.;💀+Built+by+Death+Legion+Team." alt="Footer typing"/>

</div>
