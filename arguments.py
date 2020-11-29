import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        # Dataset arguments
        self._parser.add_argument('--dataset_path', type=str, default='./data/', help='Dataset file path.')
        self._parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.')
        self._parser.add_argument('--image_size', type=int, default=64, help='Integer value for number of points.')

        # Training arguments
        self._parser.add_argument('--gpu', type=int, default=1, help='GPU number to use.')
        self._parser.add_argument('--num_epochs', type=int, default=100, help='Integer value for epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-2, help='Float value for learning rate.')
        self._parser.add_argument('--root_path', type=str, default='/mnt/hdd_10tb_1/LJJ/data/image_net/ResNet_keras/dataset/', help='Checkpoint path.')
        self._parser.add_argument('--ckpt_path', type=str, default='/mnt/hdd_10tb_1/LJJ/save/checkpoints/', help='Checkpoint path.')
        self._parser.add_argument('--result_path', type=str, default='/mnt/hdd_10tb_1/LJJ/DCGAN/save/generated/', help='Generated results path.')
        
    def parser(self):
        return self._parser