import numpy as np

class SGD(object):
    def __init__(self, parameters, momentum, lr, weight_decay):
        self.parameters = parameters
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.v = [np.zeros(p.data.shape) for p in self.parameters]

    def step(self):
        x = 1
        for i, (v, p) in enumerate(zip(self.v, self.parameters)):
            if not p.skip_decay:
                p.data -= self.weight_decay * p.data
            v = self.momentum * v + self.lr * p.grad
            self.v[i] = v
            p.data -= self.v[i]


class Adam(object):
    def __init__(self, parameters, lr, weight_decay=0, beta=(0.9, 0.999), eps=1e-8):
        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.parameters = parameters
        self.m = [np.zeros(p.data.shape) for p in self.parameters]
        self.v = [np.zeros(p.data.shape) for p in self.parameters]

        self.iterations = 0
    
    def step(self):
        self.iterations += 1
        for i, (p, m, v) in enumerate(zip(self.parameters, self.m, self.v)):
            if not p.skip_decay:
                p.data -= self.weight_decay * p.data
            m = self.beta1 * m + (1 - self.beta1) * p.grad
            v = self.beta2 * v + (1 - self.beta2) * np.power(p.grad, 2)

            self.m[i] = m
            self.v[i] = v
            
            # bias correction
            m = m / (1 - np.power(self.beta1, self.iterations))
            v = v / (1 - np.power(self.beta2, self.iterations))

            p.data -= self.lr * m / (np.sqrt(v + self.eps))

            