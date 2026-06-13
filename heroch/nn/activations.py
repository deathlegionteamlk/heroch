import numpy as np
from heroch.tensor import Tensor
from heroch.nn.module import Module

class ReLU(Module):
    def forward(self, x):
        data = np.maximum(0, x.data)
        requires_grad = x.requires_grad
        depends_on = []

        if x.requires_grad:
            def backward_func(grad):
                # dReLU(x)/dx = 1 if x > 0 else 0
                return grad * (x.data > 0).astype(np.float32)
            depends_on.append({'tensor': x, 'backward_func': backward_func})

        return Tensor(data, requires_grad, depends_on)

class Sigmoid(Module):
    def forward(self, x):
        data = 1 / (1 + np.exp(-x.data))
        requires_grad = x.requires_grad
        depends_on = []

        if x.requires_grad:
            def backward_func(grad):
                # dSigmoid(x)/dx = Sigmoid(x) * (1 - Sigmoid(x))
                return grad * (data * (1 - data))
            depends_on.append({'tensor': x, 'backward_func': backward_func})

        return Tensor(data, requires_grad, depends_on)
