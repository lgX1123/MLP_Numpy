import numpy as np
import time
from loss import *
from optimizer import *
from scheduler import *
from utils import *


class Trainer(object):
    def __init__(self, config, model=None, train_loader=None, val_loader=None):
        self.config = config
        self.epochs = self.config['epoch']
        self.lr = self.config['lr']
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.print_freq = self.config['print_freq']

        self.criterion = CrossEntropyLoss()
        if self.config['optimizer'] == 'sgd':
            self.optimizer = SGD(self.model.params, self.config['momentum'], self.lr, self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = Adam(self.model.params, self.lr)
        self.train_scheduler = CosineLR(self.optimizer, T_max=self.epochs)

    def train(self):
        best_acc1 = 0
        for epoch in range(self.epochs):
            print('current lr {:.5e}'.format(self.optimizer.lr))
            self.train_per_epoch(epoch)
            self.train_scheduler.step()

            # evaluate on validation set
            acc1 = self.validate(epoch)

            # remember best prec@1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)

    
    def train_per_epoch(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.train()

        end = time.time()

        for i, (input, target) in enumerate(self.train_loader):
            # compute output
            output = self.model.forward(input)
            loss = self.criterion(output, target)

            # compute gradient and do SGD step
            self.model.backward(self.criterion.grad)
            self.optimizer.step()

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss, input.shape[0])
            top1.update(prec1, input.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % self.print_freq == 0) or (i == len(self.train_loader) - 1):
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch + 1, i, len(self.train_loader) - 1, batch_time=batch_time,
                        loss=losses, top1=top1))
        
        output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} '.format(epoch=epoch + 1 , flag='train', top1=top1))
        print(output)
                
    def validate(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.test()

        end = time.time()
        for i, (input, target) in enumerate(self.val_loader):
            # compute output
            output = self.model.forward(input)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss, input.shape[0])
            top1.update(prec1, input.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % self.print_freq == 0) or (i == len(self.val_loader) - 1):
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(self.val_loader) - 1, batch_time=batch_time, loss=losses,
                        top1=top1))
        
        output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} '.format(epoch=epoch + 1 , flag='val', top1=top1))
        print(output)

        return top1.avg