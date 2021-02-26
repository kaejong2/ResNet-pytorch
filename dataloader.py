
import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader


from utils import *
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
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

class Custum_dataset(Dataset):
    def read_dataset(self):
        img_lst = []
        label_lst = []
        
        class_names = os.walk(self.data_path).__next__()[1]

        for i, class_name in enumerate(class_names):
            label = i
            img_dir = os.path.join(self.data_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    img_lst.append(img_file)
                    label_lst.append(label)
        return img_lst, label_lst, len(img_lst), i+1, class_names

    def __init__(self, data_path, transforms_=None, mode='train'):
        self.data_path = os.path.join(data_path, mode)
        self.transform = transforms_
        self.files, self.labels, self.len, self.num_classes, self.class_names = self.read_dataset()

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
       
        return {'img': img, 'label': self.labels[index]}

    def __len__(self):
        return self.len

def data_loader(args, mode="train"):
    data_path = os.path.join(args.root_path, args.data_path)
    if mode=='train':
        transforms_ = transforms.Compose(
            [transforms.Resize((224, 224)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
        dataset = Custum_dataset(data_path, transforms_=transforms_, mode = mode)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    else:
        transforms_ = transforms.Compose(
            [transforms.Resize((224, 224), Image.BICUBIC), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = Custum_dataset(data_path, transforms_=transforms_, mode = mode)

        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    return dataset, dataloader