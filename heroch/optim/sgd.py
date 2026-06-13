from heroch.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [0.0 for _ in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + (1 - self.momentum) * p.grad
                p.data -= self.lr * self.velocities[i]
            else:
                p.data -= self.lr * p.grad
