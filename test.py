import os, sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from utils import *
from dataloader import *
from model import *
import time
from torchvision import transforms, models
def test(args):
    start_time = time.time()
    # net = ResNet(in_channels=args.input_nc, num_classes=args.num_classes, resblock=[3, 4, 6, 3]).to(device=args.device)
    net = models.resnet34()
    num_features= net.fc.in_features
    net.fc = nn.Linear(num_features, 3)
    net = net.to(device=args.device)
    try:
        ckpt = load_checkpoint(args, args.device)
        net.load_state_dict(ckpt['Model'])
    except:
        print('Failed to load checkpoint')

    test_datasets, test_dataloader = data_loader(args, mode = 'train')

    net.eval()
    start_time = time.time()
    class_names = test_datasets.class_names
    with torch.no_grad():
        losses = 0.
        corrects = 0

        for _iter, data in enumerate(test_dataloader):
            inputs = data['img'].to(device=args.device)
            labels = data['label'].to(device=args.device)

            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)

            corrects += torch.sum(preds == labels.data)

            print(f'[Predicted Result: {class_names[preds[0]]}] (Ground Truth: {class_names[labels.data[0]]})')
            # imshow(inputs.cpu().data[0], title=' Predicted Result: ' + class_names[preds[0]])
        epoch_acc = corrects / len(test_datasets) * 100.
        print('[Test Phase]  Acc: {:.4f}% Time: {:.4f}s'.format(epoch_acc, time.time() - start_time))

