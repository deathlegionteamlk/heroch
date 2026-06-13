import numpy as np
from heroch.tensor import Tensor
from heroch.nn.module import Module

class MSELoss(Module):
    def forward(self, input, target):
        # input: Tensor (batch_size, ...)
        # target: Tensor (batch_size, ...)
        diff = input - target
        # Mean squared error: sum((input - target)^2) / n
        n = np.prod(input.shape)
        
        # We need a pow operation for Tensor, or just use mul
        sq_diff = diff * diff
        loss = sq_diff.sum() * (1.0 / n)
        
        return loss
