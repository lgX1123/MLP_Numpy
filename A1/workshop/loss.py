import numpy as np
from layers import softmax

class CrossEntropyLoss(object):
    def __init__(self):
        self.softmax = softmax('softmax')
    
    def gradient(self):
        return self.grad

    def __call__(self, input, ground_truth):
        preds = self.softmax.forward(input)

        self.grad = preds - ground_truth    #TODO: 推导要写在report上不？

        bacth_size = input.shape[0]
        loss = -1 * (ground_truth * np.log(preds)).sum() / bacth_size

        return loss