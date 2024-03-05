import numpy as np

class SGD(object):
    def __init__(self, parameters, momentum, lr, weight_decay):
        self.parameters = parameters
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.v = [np.zeros(p.data.shape) for p in self.parameters]

    def step(self):
        for i, (v, p) in enumerate(zip(self.v, self.parameters)):
            if not p.skip_decay:
                p.data -= self.weight_decay * p.data
            v = self.momentum * v + self.lr * p.grad
            self.v[i] = v
            p.data -= self.v[i]