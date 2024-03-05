import numpy as np
from layers import softmax

class CrossEntropyLoss(object):
    def __init__(self):
        self.softmax = softmax('softmax')

    def __call__(self, input, ground_truth):
        self.bacth_size = input.shape[0]
        self.class_num = input.shape[1]

        preds = self.softmax.forward(input)
        ground_truth = self.one_hot_encoding(ground_truth)

        self.grad = preds - ground_truth    #TODO: 推导要写在report上不？

        loss = -1 * (ground_truth * np.log(preds)).sum() / self.bacth_size

        return loss
    
    def one_hot_encoding(self, x):
        one_hot_encoded = np.zeros((self.bacth_size, self.class_num))
        one_hot_encoded[np.arange(x.size), x.flatten()] = 1
        return one_hot_encoded