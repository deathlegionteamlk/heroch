import numpy as np
from heroch.tensor import Tensor
from heroch.nn.module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32),
            requires_grad=True
        )
        if bias:
            self.bias = Tensor(
                np.zeros((1, out_features), dtype=np.float32),
                requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x):
        # x is (batch_size, in_features)
        # weight is (in_features, out_features)
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
