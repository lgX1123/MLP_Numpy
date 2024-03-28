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

        self.accu_beta1 = self.beta1
        self.accu_beta2 = self.beta2
    
    def step(self):
        self.accu_beta1 *= self.beta1
        self.accu_beta2 *= self.beta2
        # lr = self.lr * ((1 - self.accu_beta2) ** 0.5) / (1 -self.accu_beta1)
        lr = self.lr
        for i, (p, m, v) in enumerate(zip(self.parameters, self.m, self.v)):
            if not p.skip_decay:
                p.data *= self.weight_decay
            self.m[i] = self.beta1 * m + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * v + (1 - self.beta2) * np.power(p.grad, 2)
            
            # bias correction
            #TODO

            p.data -= lr * self.m[i] / (np.sqrt(self.v[i] + self.eps))

            