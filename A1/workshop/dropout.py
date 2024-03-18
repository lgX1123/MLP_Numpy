import numpy as np
from layers import Layer


class dropout(Layer):
    def __init__(self, name, drop_rate, requires_grad=False):
        super().__init__(name, requires_grad)
        self.drop_rate = drop_rate
        self.fix_value = 1 / (1 - self.drop_rate)   # to keep average fixed

    def forward(self, input):
        if self.train:
            self.mask = np.random.uniform(0, 1, input.shape) > self.drop_rate
            return input * self.mask * self.fix_value
        else:
            return input

    def backward(self, grad_output):
        if self.train:
            return grad_output * self.mask
        else:
            return grad_output