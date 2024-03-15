from layers import Layer
from utils import Parameter
import numpy as np

class batchnorm(Layer):
    def __init__(self, name, shape, requires_grad=True):
        super().__init__(name)
        self.gamma = Parameter(np.random.uniform(0.9, 1.1, shape), requires_grad, skip_decay=True)
        self.beta = Parameter(np.random.uniform(-0.1, 0.1, shape), requires_grad, skip_decay=True)
        self.requires_grad = requires_grad

        self.overall_mean = Parameter(np.zeros(shape), False)
        self.overall_var = Parameter(np.zeros(shape), False)

    
    def forward(self, input):
        if self.train:
            batch_mean = input.mean(axis=0)
            batch_std = np.sqrt(input.std(axis=0) + 1e-8)    # To avoid divided by 0
            
            self.norm = (input - batch_mean) / batch_std
            self.gamma_norm = self.gamma.data / batch_std

            # compute overall mean and var. Estimate, no need use accurate formula which may cause overflow.


            return self.gamma.data * self.norm + self.beta.data
        
        else:
            pass


    
    def backward(self, grad_output):        
        batch_size = grad_output.shape[0]
        self.gamma.grad = (grad_output * self.norm).sum(axis=0) / batch_size
        self.beta.grad = grad_output.sum(axis=0) / batch_size
        return self.gamma_norm * (grad_output - self.norm * self.gamma.grad - self.beta.grad)       # TODO: 推导