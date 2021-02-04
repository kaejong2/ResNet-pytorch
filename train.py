import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from utils import *

from dataloader import MNIST_loader, CIFAR10_loader
from model import ResNet

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std: (x * std) + mean


class Classification():
    def __init__(self, args):
        self.args = args 
        #dataset
        self.train_data, self.test_data = CIFAR10_loader(self.args)
        #Module
        self.net = ResNet(in_channels=self.args.input_nc,num_classes=self.args.num_classes,resblock=[2, 2, 2, 2]).to(device=self.args.device)
        init_weight(self.net, init_type="kaiming", init_gain=0.02)
        self.criterion = nn.CrossEntropyLoss().to(device=self.args.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum = 0.9, weight_decay=5e-4)
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)

    def run(self, ckpt_path=None, result_path=None):
        pbar_epoch = tqdm(total=self.args.num_epochs, desc='Epoch', position=0)
        for epoch in range(self.args.num_epochs):
            pbar_iter = tqdm(total=len(self.train_data), desc='Batch', position=1)
            train_loss = 0.0
            correct_pred = 0
            total =0
            self.net.train()
            
            for _iter, (data, label) in enumerate(self.train_data):
                data = data.to(device=self.args.device)
                label = label.to(device=self.args.device)

                output = self.net(data)

                loss = self.criterion(output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct_pred += predicted.eq(label).sum().item()
                if _iter % 50 ==49:
                    pbar_iter.write('[%d, %5d] loss: %.3f Acc: %.3f%% (%d/%d)'  %
                  (epoch, _iter + 1, loss.item(),100.*correct_pred/total ,correct_pred, total))
                pbar_iter.update()
            save(ckpt_path, self.net, self.optimizer, epoch)
            pbar_epoch.update()
            self.lr_scheduler.step()
            print(self.lr_scheduler.optimizer.state_dict()['param_groups'][0]['lr'])
            

    def test(self, load_ckpt=None, result_path=None):
        self.net, self.optimizer, epoch = load(self.args, load_ckpt, self.net, self.optimizer, epoch=1)
        self.net.eval()
        test_loss = []
        total = 0
        correct_pred = 0
        for _iter, (data, label) in enumerate(self.test_data):
            data = data.to(device=self.args.device)
            label = label.to(device=self.args.device)

            output = self.net(data)

            loss = self.criterion(output, label)
            _, predicted = output.max(1)
            total += label.size(0)
            correct_pred += predicted.eq(label).sum().item()
            test_loss += [loss.item()]
            print("Test: Batch %3d / %3d | Loss %.4f | Acc %.4f" %(_iter, len(self.test_data), np.mean(test_loss), 100.*correct_pred/total))

            # label = label.to('cpu').detach().numpy()
            # data = fn_tonumpy(fn_denorm(data, mean=0.5, std=0.5))
            # predicted = predicted.to('cpu').detach().numpy()

            # for j in range(label.shape[0]): # batch의 각 데이터 class 분류
            #     id = self.args.batch_size * (_iter) + j

            #     label_ = label[j]
            #     data_ = data[j]
            #     predicted_ = predicted[j]

            #     data_ = np.clip(data_, a_min=0, a_max=1)

                # plt.imsave(os.path.join(result_path, 'png', '%d_data.png' % id), data_, cmap=None)


