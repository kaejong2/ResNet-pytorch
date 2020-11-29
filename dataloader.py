import torch
import torchvision
import torchvision.transforms as transforms
import glob
def MNIST(args):
    # Image processing
    transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,),(0.5,))])

    # MNIST dataset
    mnist = torchvision.datasets.MNIST(root=args.root_path,
                                    train=True,
                                    transform=transform,
                                    download=True)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                            batch_size=args.batch_size, 
                                            shuffle=True)

    return data_loader

def CIFAR10(args):
    # Image processing
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.root_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=args.root_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


def Imagefolder(args):
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.ImageFolder(root=args.root_path+"train", transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=args.root_path+"test", transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

# class Custum_data(args):
#     def __init__(self, args, root, npoint):
#         super(Custum_data,self).__init__()
#         self.file_list = glob.glob(args.root_path)


#     def __getitem__(self, index):
#         fn = self.datapath[index]

#         point_set = genfromtxt(fn, delimiter=',').astype(np.float32)
        
#         choice = np.random.choice(len(point_set), size=4096, replace=False)

#         point_set = point_set[choice]

#         point_set = torch.from_numpy(point_set)
#         point_set = point_set/512
#         return point_set

#     def __len__(self):
#         return len(self.datapath)

