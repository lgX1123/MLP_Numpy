import numpy as np
import time
import math

from dataloader import Dataloader
from MLP import MLP
from layers import *
from Trainer import Trainer
from batchnorm import batchnorm

def get_model(layers):
    model = MLP()
    str2obj = {
        'linear': HiddenLayer, 
        'relu': relu, 
        'sigmoid': sigmoid, 
        'batchnorm': batchnorm,
    }
    for i in layers:
        model.add_layer(str2obj[i['type']](**i['params']))

    return model

@timer
def main():
    file_path = '../Assignment1-Dataset/'

    train_X = np.load(file_path + 'train_data.npy')
    train_y = np.load(file_path + 'train_label.npy')
    test_X = np.load(file_path + 'test_data.npy')
    test_y = np.load(file_path + 'test_label.npy')

    layers = [
        {'type': 'linear', 'params': {'name': 'fc1', 'in_num': 128, 'out_num': 64}},
        {'type': 'batchnorm', 'params': {'name': 'bn1', 'shape': 64}}, 
        {'type': 'relu', 'params': {'name': 'relu1'}}, 
        {'type': 'linear', 'params': {'name': 'fc2', 'in_num': 64, 'out_num': 10}}
    ]
    lr = 0.001
    bs = 1024
    momentum = 0.9
    weight_decay = 5e-3     # 2e-4, 1e-4
    seed = 0
    epoch = 10

    config = {
        'layers': layers,
        'lr': lr, 
        'bs': bs,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'seed': seed,
        'epoch': epoch,
        
    }


    train_dataloader = Dataloader(train_X, train_y, config['bs'], shuffle=True, seed=config['seed'])
    test_dataloader = Dataloader(test_X, test_y, config['bs'], shuffle=False)
    model = get_model(config['layers'])
    trainer = Trainer(config, model, train_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()