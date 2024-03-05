from layers import Layer
from utils import Parameter
import numpy as np

class batchnorm(Layer):
    def __init__(self, name, shape, requires_grad=True):
        super().__init__(name)
        self.W = Parameter(np.random.uniform(0.9, 1.1, shape), requires_grad, skip_decay=True)
        self.b = Parameter(np.random.uniform(-0.1, 0.1, shape), requires_grad, skip_decay=True)
        self.requires_grad = requires_grad

    
    def forward(self, input):
        batch_mean = input.mean(axis=0)
        batch_std = np.sqrt(input.std(axis=0) + 1e-8)    # To avoid divided by 0
        
        self.norm = (input - batch_mean) / batch_std
        self.gamma_norm = self.W.data / batch_std

        return self.W.data * self.norm + self.b.data
    
    def backward(self, grad_output):        # TODO: 推导
        batch_size = grad_output.shape[0]
        self.W.grad = (grad_output * self.norm).sum(axis=0) / batch_size
        self.b.grad = grad_output.sum(axis=0) / batch_size
        return self.gamma_norm * (grad_output - self.norm * self.W.grad - self.b.grad)       