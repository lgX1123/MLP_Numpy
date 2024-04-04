import numpy as np
import time
import math

from dataloader import Dataloader
from MLP import MLP
from layers import *
from Trainer import Trainer
from batchnorm import batchnorm
from dropout import dropout
from utils import *



def get_model(layers):
    model = MLP()
    str2obj = {
        'linear': HiddenLayer, 
        'relu': relu, 
        'leaky_relu': leaky_relu,
        'sigmoid': sigmoid, 
        'tanh': tanh,
        'batchnorm': batchnorm,
        'dropout': dropout
    }
    for i in layers:
        model.add_layer(str2obj[i['type']](**i['params']))
    
    return model


def main():
    file_path = '../Assignment1-Dataset/'

    train_X = np.load(file_path + 'train_data.npy')
    train_y = np.load(file_path + 'train_label.npy')
    test_X = np.load(file_path + 'test_data.npy')
    test_y = np.load(file_path + 'test_label.npy')

    layers = [
        {'type': 'linear', 'params': {'name': 'fc1', 'in_num': 128, 'out_num': 64}},
        {'type': 'batchnorm', 'params': {'name': 'bn1', 'shape': 64}}, 
        {'type': 'dropout', 'params': {'name': 'dropout1', 'drop_rate': 0.1}},
        # {'type': 'sigmoid', 'params': {'name': 'sigmoid'}},  
        {'type': 'leaky_relu', 'params': {'name': 'leaky_relu1', 'alpha': 0.1}},  
        #{'type': 'relu', 'params': {'name': 'relu1'}},  
        #{'type': 'tanh', 'params': {'name': 'tanh1'}},  
        {'type': 'linear', 'params': {'name': 'fc2', 'in_num': 64, 'out_num': 32}},
        {'type': 'batchnorm', 'params': {'name': 'bn2', 'shape': 32}}, 
        {'type': 'dropout', 'params': {'name': 'dropout2', 'drop_rate': 0.1}},
        {'type': 'relu', 'params': {'name': 'relu2'}}, 
        {'type': 'linear', 'params': {'name': 'fc3', 'in_num': 32, 'out_num': 10}},
    ]
  
    bs = 128
    config = {
        'layers': layers,
        'lr': 0.1, 
        'bs': bs,
        'momentum': 0.9,
        'weight_decay': 5e-4,   # 5e-4, 2e-4, 1e-4, 5e-3, 0
        'seed': 0,
        'epoch': 100,
        'optimizer': 'sgd',  # adam, sgd
        'pre-process': 'norm',      # min-max, norm, None
        'print_freq': 50000 // bs // 5
    }
    np.random.seed(config['seed'])

    # pre process
    train_X, test_X = get_transform(train_X, test_X, config['pre-process'])

    train_dataloader = Dataloader(train_X, train_y, config['bs'], shuffle=True, seed=config['seed'])
    test_dataloader = Dataloader(test_X, test_y, config['bs'], shuffle=False)
    model = get_model(config['layers'])
    trainer = Trainer(config, model, train_dataloader, test_dataloader)
    trainer.train()
    trainer.plot_cm('../figs/cm.png')


if __name__ == '__main__':
    main()