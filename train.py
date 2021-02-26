import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torchvision.utils import save_image
from torch.autograd import Variable

from torchvision import datasets
import torchvision.transforms as transforms

from utils import *

from dataloader import *
from model import ResNet

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std: (x * std) + mean


class Classification():
    def __init__(self, args):
        self.args = args 

        #dataset
        self.train_datasets, self.train_dataloader = data_loader(self.args, mode="train")
        self.test_datasets, self.test_dataloader = data_loader(self.args, mode='test')

        # Network
        # self.net = ResNet(in_channels=self.args.input_nc,num_classes=self.args.num_classes,resblock=[3, 4, 6, 3]).to(device=self.args.device)
        # init_weight(self.net, init_type="kaiming", init_gain=0.02)

        self.net = models.resnet34(pretrained=True)
        num_features= self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, 3)
        self.net = self.net.to(device=self.args.device)

        self.criterion = nn.CrossEntropyLoss().to(device=self.args.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum = 0.9, weight_decay=5e-4)
        
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)

    def run(self):
        start_time = time.time()
        self.net.train()
        for epoch in range(self.args.num_epochs):
            losses = 0.
            corrects = 0
            
            for _iter, inputs in enumerate(self.train_dataloader):
                data = inputs['img'].to(device=self.args.device)
                label = inputs['label'].to(device=self.args.device)

                self.optimizer.zero_grad()
                output = self.net(data)
                _, predicted = torch.max(output, 1)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                

                losses += loss.item() * data.size(0)
                corrects += torch.sum(predicted==label.data)

            epoch_loss = losses / len(self.train_datasets)
            epoch_acc = corrects / len(self.train_datasets) * 100.
            
            print("Train' Epoch: %d | Loss: %.4f | Acc: %.4f Time: %.4f" %(epoch, epoch_loss, epoch_acc, time.time()-start_time))
            if epoch % 10 ==9:
                save(os.path.join(self.args.root_path, self.args.ckpt_path), self.net, self.optimizer, epoch)

            # self.lr_scheduler.step()
            # print(self.lr_scheduler.optimizer.state_dict()['param_groups'][0]['lr'])
            

            self.net.eval()
            start_time = time.time()
            class_names = self.test_datasets.class_names
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for _iter, data in enumerate(self.test_dataloader):
                inputs = data['img'].to(device=self.args.device)
                labels = data['label'].to(device=self.args.device)

                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                print(f'[Predicted Result: {class_names[preds[0]]}] (Ground Truth: {class_names[labels.data[0]]})')
                # imshow(inputs.cpu().data[0], title=' Predicted Result: ' + class_names[preds[0]])

            epoch_loss = running_loss / len(self.test_datasets)
            epoch_acc = running_corrects / len(self.test_datasets) * 100.
            print('[Test Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc, time.time() - start_time))