import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def get_array_module(data):
    if HAS_CUPY:
        return cp.get_array_module(data)
    return np

class Tensor:
    def __init__(self, data, requires_grad=False, depends_on=None):
        xp = cp if HAS_CUPY and isinstance(data, cp.ndarray) else np
        
        if isinstance(data, (int, float)):
            data = xp.array(data, dtype=np.float32)
        elif isinstance(data, list):
            data = xp.array(data, dtype=np.float32)
        
        self.data = data
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.grad = None
        self.shape = self.data.shape

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self):
        xp = get_array_module(self.data)
        self.grad = xp.zeros_like(self.data, dtype=np.float32)

    def to_gpu(self):
        if HAS_CUPY:
            self.data = cp.asarray(self.data)
            if self.grad is not None:
                self.grad = cp.asarray(self.grad)
        return self

    def to_cpu(self):
        if HAS_CUPY and isinstance(self.data, cp.ndarray):
            self.data = cp.asnumpy(self.data)
            if self.grad is not None:
                self.grad = cp.asnumpy(self.grad)
        return self

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad=None):
        xp = get_array_module(self.data)
        if grad is None:
            if self.shape == ():
                grad = xp.array(1.0, dtype=np.float32)
            else:
                raise RuntimeError("backward() can only be called on scalar tensors if no gradient is provided.")
        
        if isinstance(grad, Tensor):
            grad = grad.data

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        for dependency in self.depends_on:
            backward_func = dependency['backward_func']
            tensor = dependency['tensor']
            grad_to_propagate = backward_func(grad)
            tensor.backward(grad_to_propagate)

    def __add__(self, other):
        other = ensure_tensor(other)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        depends_on = []

        if self.requires_grad:
            def backward_func(grad):
                # Gradient of sum is just the gradient (with broadcast handling)
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad
            depends_on.append({'tensor': self, 'backward_func': backward_func})
        
        if other.requires_grad:
            def backward_func(grad):
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad
            depends_on.append({'tensor': other, 'backward_func': backward_func})

        return Tensor(data, requires_grad, depends_on)

    def __mul__(self, other):
        other = ensure_tensor(other)
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        depends_on = []

        if self.requires_grad:
            def backward_func(grad):
                # d(x*y)/dx = y
                res = grad * other.data
                ndims_added = res.ndim - self.data.ndim
                for _ in range(ndims_added):
                    res = res.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        res = res.sum(axis=i, keepdims=True)
                return res
            depends_on.append({'tensor': self, 'backward_func': backward_func})

        if other.requires_grad:
            def backward_func(grad):
                # d(x*y)/dy = x
                res = grad * self.data
                ndims_added = res.ndim - other.data.ndim
                for _ in range(ndims_added):
                    res = res.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        res = res.sum(axis=i, keepdims=True)
                return res
            depends_on.append({'tensor': other, 'backward_func': backward_func})

        return Tensor(data, requires_grad, depends_on)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * (-1.0)

    def __matmul__(self, other):
        # self @ other
        other = ensure_tensor(other)
        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        depends_on = []

        if self.requires_grad:
            def backward_func(grad):
                # d(X@W)/dX = grad @ W.T
                return grad @ other.data.T
            depends_on.append({'tensor': self, 'backward_func': backward_func})
        
        if other.requires_grad:
            def backward_func(grad):
                # d(X@W)/dW = X.T @ grad
                return self.data.T @ grad
            depends_on.append({'tensor': other, 'backward_func': backward_func})
        
        return Tensor(data, requires_grad, depends_on)

    def sum(self):
        data = self.data.sum()
        requires_grad = self.requires_grad
        depends_on = []

        if self.requires_grad:
            def backward_func(grad):
                # Gradient of sum is just the gradient broadcasted to original shape
                xp = get_array_module(self.data)
                return grad * xp.ones_like(self.data)
            depends_on.append({'tensor': self, 'backward_func': backward_func})

        return Tensor(data, requires_grad, depends_on)

    @property
    def T(self):
        data = self.data.T
        requires_grad = self.requires_grad
        depends_on = []
        if self.requires_grad:
            def backward_func(grad):
                return grad.T
            depends_on.append({'tensor': self, 'backward_func': backward_func})
        return Tensor(data, requires_grad, depends_on)

def ensure_tensor(obj):
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)
