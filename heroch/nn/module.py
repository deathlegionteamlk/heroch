from heroch.tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = {}

    def parameters(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Tensor) and item.requires_grad:
                        yield item
                    elif isinstance(item, Module):
                        yield from item.parameters()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
