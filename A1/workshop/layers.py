import numpy as np
import math

from utils import *

class Layer(object):
    def __init__(self, name, requires_grad=False):
        self.name = name 
        self.requires_grad = requires_grad
        self.train = True
        
    def forward(self, *args):
        pass

    def backward(self, *args):
        pass


class relu(Layer):
    def __init__(self, name, requires_grad=False):
        super().__init__(name, requires_grad)

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, grad_output):
        grad_output[self.input <= 0] = 0
        return grad_output
    

class sigmoid(Layer):
    def __init__(self, name, requires_grad=False):
        super().__init__(name, requires_grad)
        
    def forward(self, input):
        self.y = 1. / (1. + np.exp(-input))   # save sigmoid for more convenient grad computation
        return self.y
    
    def backward(self, grad_output):
        return self.y * (1 - self.y) * grad_output
    

class softmax(Layer):
    def __init__(self, name, requires_grad=False):
        super().__init__(name, requires_grad)
        
    def forward(self, input):
        """
            input.shape = [batch size, num_class]
        """
        x_max = input.max(axis=-1, keepdims=True)       # to avoid overflow
        x_exp = np.exp(input - x_max)
        return x_exp / x_exp.sum(axis=-1, keepdims=True)
    
    def backward(self, grad_output):
        # packaged in CrossEntropyLoss
        pass


class HiddenLayer(Layer):
    def __init__(self, name, in_num, out_num):
        super().__init__(name, requires_grad=True)
        self.in_num = in_num
        self.out_num = out_num

        W = kaiming_normal_(np.array([0] * in_num * out_num).reshape(in_num, out_num), a=math.sqrt(5))     # Kaiming Init
        self.W = Parameter(W, self.requires_grad)
        self.b = Parameter(np.zeros(out_num), self.requires_grad)

    def forward(self, input):
        self.input = input
        return input @ self.W.data + self.b.data      # [batch size, in_num] @ [in_num, out_num] + [out_num] => [batch size, out_num]
    
    def backward(self, grad_output):
        """
            grad_output: [batch size, out_num]
        """
        batch_size = grad_output.shape[0]
        self.W.grad = self.input.T @ grad_output / batch_size
        self.b.grad = grad_output.sum(axis=0) / batch_size
        return grad_output @ self.W.data.T
    
