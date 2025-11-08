import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
ROOT_DIR = r"C:\Users\delga\Documents\programming\datasets"

# cifar10_dir = os.path.join(ROOT_DIR, "cifar-10-batches-py")         #CIFAR10
# if not os.path.exists(cifar10_dir):
#     print("Downloading CIFAR10...")
#     trainCIFAR10 = datasets.CIFAR10(root=ROOT_DIR, train=True, download=True)
#     testCIFAR10 = datasets.CIFAR10(root=ROOT_DIR, train=False, download=True)
# else:
#     print("CIFAR10 already exists, loading without download...")
#     trainCIFAR10 = datasets.CIFAR10(root=ROOT_DIR, train=True, download=False)
#     testCIFAR10 = datasets.CIFAR10(root=ROOT_DIR, train=False, download=False)


# mnist_dir = os.path.join(ROOT_DIR, "MNIST")             #MNIST
# if not os.path.exists(mnist_dir):
#     print("Downloading MNIST...")
#     trainMNIST = datasets.MNIST(root=ROOT_DIR, train=True, download=True)
#     testMNIST = datasets.MNIST(root=ROOT_DIR, train=False, download=True)
# else:
#     print("MNIST already exists, loading without download...")
#     trainMNIST = datasets.MNIST(root=ROOT_DIR, train=True, download=False)
#     testMNIST = datasets.MNIST(root=ROOT_DIR, train=False, download=False)

def get_dataloaders(batch_size=128):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])

    dataset = datasets.CIFAR10(root=ROOT_DIR, train=True, download=False, transform=train_transforms)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    test_dataset = datasets.CIFAR10(root=ROOT_DIR, train=False, download=False, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

