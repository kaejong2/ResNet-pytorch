import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
import torchvision.utils as vutils
from dataloader import CIFAR10, Imagefolder
from arguments import Arguments
from model.VGGNet import VGG
from utils import progress_bar
from torchvision import datasets, models, transforms

class Main():
    def __init__(self,args):
        self.args = args
        #dataset

        self.trainset, self.testset = Imagefolder(self.args)
        #Module
        # self.net = VGG().to(device= args.device)
        self.model_ft = models.resnet18(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        # 여기서 각 출력 샘플의 크기는 2로 설정합니다.
        # 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
        self.model_ft.fc = nn.Linear(num_ftrs, 20)

        self.model_ft = self.model_ft.to(device= args.device)
        # self.net.load_state_dict(torch.load(self.args.ckpt_path+"Resnet_epoch_60.pth"))
        # self.net = self.net
        self.criterion = nn.CrossEntropyLoss().to(device= args.device)
        self.optimizer = optim.SGD(self.model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)

    def run(self):
        
        print('\nEpoch: %d' % epoch)
        self.model_ft.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainset):
            inputs, targets = inputs.to(device=self.args.device), targets.to(device=self.args.device)
            self.optimizer.zero_grad()
            outputs = self.model_ft(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # if epoch%10==0:
        #     # torch.save(self.net.state_dict(), '%sResnet_epoch_%d.pth' % (save_ckpt, epoch))
        self.scheduler.step()
            
    def test(self, save_ckpt=None):
        
        global best_acc
        self.model_ft.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testset):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.model_ft(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            
            torch.save(self.model_ft.state_dict(), '%sResnet_epoch_%d.pth' % (save_ckpt, epoch))
            best_acc = acc


                
if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Create a directory if not exists
    best_acc = 0
    if os.path.exists(args.ckpt_path) is False:
        os.makedirs(args.ckpt_path)

    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)

    
    model = Main(args)
    for epoch in range(args.num_epochs):
        model.run()
        model.test(save_ckpt=args.ckpt_path)