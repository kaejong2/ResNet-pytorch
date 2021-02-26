import argparse
import torch
from train import Classification
import os
from crawling import *
from test import test
def Arguments():
    parser = argparse.ArgumentParser(description='Arguments for pix2pix.')

    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
    parser.add_argument('--mode', type=str, default='test', choices=["train", "test"], help='Run type.')
    # Dataset arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Integer value for batch size.')
    parser.add_argument('--image_size', type=int, default=224, help='Integer value for number of points.')
    parser.add_argument('--input_nc', type=int, default=3, help='size of image height')
    parser.add_argument('--num_classes', type=int, default=3, help='size of image height')
    
    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=0.001, help='Adam : learning rate.')
    parser.add_argument('--momentum', type=int, default=0.9, help="Momentum")
    parser.add_argument('--weight_decay', type=int, default=5e-4, help="epoch from which to start lr decay")
    # Training arguments
    parser.add_argument('--epoch', type=int, default=0, help='Epoch to start training from.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs of training.')
    parser.add_argument('--root_path', type=str, default="../resnet", help='Root path.')
    parser.add_argument('--data_path', type=str, default='data/celeb', help='Data path.')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/celeb', help='Checkpoint path.')
    parser.add_argument('--load_path', type=str, default='ckpt/celeb', help='Checkpoint path.')
    parser.add_argument('--result_path', type=str, default='result/celeb', help='Results path.')
    
    # Network argument
    parser.add_argument('--model_scope', type=str, default="resnet", help='')
    parser.add_argument('--resblock', type=str, default="2222", help='Enter 4 digits')
    parser.add_argument('--pretrained', type=str, default=False, help='')
    parser.add_argument('--init_weight', type=str, default="kaiming", help='')

    
    # Model arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = Arguments()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    # os.makedirs("%s/%s" % (args.root_path, args.data_path), exist_ok=True)
    # os.makedirs("%s/%s" % (args.root_path, args.ckpt_path), exist_ok=True)
    # os.makedirs("%s/%s" % (args.root_path, args.result_path), exist_ok=True)

    # for arg, value in vars(args).items():
    #     print("log %s : %r" % (arg, value))
    
    # if args.mode == 'train':
    #     model = Classification(args)
    #     model.run()
    # elif args.mode == 'test':
    #     os.makedirs("%s/%s_test" % (args.root_path, args.result_path), exist_ok=True)
    #     test(args)
    # elif args.mode == 'data_crawling':
    #     query = 'Jake Gyllenhaal'
    #     data_crawler(args, query)
    block = [int(args.resblock[0]), int(args.resblock[1]),int(args.resblock[2]),int(args.resblock[3])]