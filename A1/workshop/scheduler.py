import numpy as np

class CosineLR(object):
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self.n = -1
        self.base_lr = optimizer.lr
        self.step()

    def step(self):
        self.n += 1
        lr = self.get_lr()
        self.optimizer.lr = lr

    def get_lr(self):
        cos = np.cos(np.pi * self.n / self.T_max)
        return self.base_lr * (1 + cos) / 2