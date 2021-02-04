
import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch
import torchvision
import torchvision.transforms as transforms



def MNIST_loader(args):
    # Image processing
    transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,),(0.5,))])

    # MNIST Data loader
    train_loader = torchvision.datasets.MNIST(root=args.data_path, transform=transform, train=True, download=True)
    val_loader = torchvision.datasets.MNIST(root=args.data_path, transform=transform, train=False, download=True)
    test_loader = torchvision.datasets.MNIST(root=args.data_path, transform=transform, train=False, download=True)

    trainloader = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,shuffle=False)
    # return train_loader, val_loader, test_loader
    return trainloader ,testloader

def CIFAR10_loader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, transform=transform, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, transform=transform, train=False, download=True)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader