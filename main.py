import argparse
import torch
from train import Classification

def Arguments():
    parser = argparse.ArgumentParser(description='Arguments for pix2pix.')

    parser.add_argument('--gpu', type=int, default=1, help='GPU number to use.')
    # Dataset arguments
    parser.add_argument('--batch_size', type=int, default=256, help='Integer value for batch size.')
    parser.add_argument('--image_size', type=int, default=256, help='Integer value for number of points.')
    parser.add_argument('--input_nc', type=int, default=3, help='size of image height')
    parser.add_argument('--num_classes', type=int, default=10, help='size of image height')
    
    # Optimizer arguments
    parser.add_argument('--b1', type=int, default=0.5, help='GPU number to use.')
    parser.add_argument('--b2', type=int, default=0.999, help='GPU number to use.')
    parser.add_argument('--lr', type=float, default=0.1, help='Adam : learning rate.')
    parser.add_argument('--decay_epoch', type=int, default=20, help="epoch from which to start lr decay")
    # Training arguments
    parser.add_argument('--epoch', type=int, default=0, help='Epoch to start training from.')
    parser.add_argument('--num_epochs', type=int, default=120, help='Number of epochs of training.')
    parser.add_argument('--data_path', type=str, default='/mnt/hdd/LJJ/resnet/', help='Checkpoint path.')
    parser.add_argument('--ckpt_path', type=str, default='/mnt/hdd/LJJ/resnet/ckpt/', help='Checkpoint path.')
    parser.add_argument('--log_path', type=str, default='/mnt/hdd/LJJ/resnet/log/', help='Generated results path.')
    parser.add_argument('--result_path', type=str, default='/mnt/hdd/LJJ/resnet/result/', help='Generated results path.')
    
    # Network argument
    
    # Model arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = Arguments()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    model = Classification(args)
    
    model.run(ckpt_path=args.ckpt_path, result_path=args.result_path)
 
