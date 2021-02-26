import random
import time
import datetime
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import init
import shutil
from model import *
from torchvision import transforms, models


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def init_weight(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def save(ckpt_dir, Model, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'Model' : Model.state_dict(), 'optim': optim.state_dict()}, "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load_checkpoint(args, device):
    ckpt_lst = os.listdir(os.path.join(args.root_path, args.load_path))
    ckpt_lst.sort()
    dict_model = torch.load('%s/%s'% (os.path.join(args.root_path, args.load_path), ckpt_lst[-1]), map_location=device)
    print('Loading checkpoint from %s/%s succeed' % (args.load_path, ckpt_lst[-1]))
    return dict_model

def dataset_split(query, train_cnt, directory_lst):
    for dir in directory_lst:
        if not os.path.isdir(dir+"/"+query):
            os.makedirs(dir+"/"+query)
    cnt = 0
    for file_name in os.listdir(query):
        if cnt < train_cnt:
            print(f'[Train dataset] {file_name}')
            shutil.move(query + '/' + file_name, directory_lst[0] + query + '/' + file_name)
        else:
            print(f'[Test dataset] {file_name}')
            shutil.move(query + '/' + file_name, directory_lst[1] + query + '/' + file_name)
        cnt += 1
    shutil.rmtree(query)

def imshow(input, title):
    input = input.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input,0,1)

    plt.imshow(input)
    plt.title(title)
    plt.show()

def model_select(args):
    if args.model_scope == "resnet":
        resblock = [int(args.resblock[0]), int(args.resblock[1]),int(args.resblock[2]),int(args.resblock[3])]
        model = ResNet(in_channels=args.input_nc, num_classes=args.num_classes, resblock=resblock).to(device=args.device)
        init_weight(model, init_type=args.init_weight, init_gain=0.02)
    
    elif args.model_scope == "VGG":
        model = VGG(in_channels=args.input_nc, out_channels=args.num_classes).to(device=args.device)
        init_weight(model, init_type=args.init_weight, init_gain=0.02)
    
    elif args.model_scope == "VGG":
        model = LeNet5(n_classes=args.num_classes).to(device=args.device)
        init_weight(model, init_type=args.init_weight, init_gain=0.02)

    elif args.model_scope == "pre_resnet":
        model = models.resnet34(pretrained=True)
        num_features= model.fc.in_features
        model.fc = nn.Linear(num_features, args.num_classes)
        model = model.to(device=args.device)

    return model