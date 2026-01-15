import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from dotenv import load_dotenv
load_dotenv()


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing environment variable: {name}")
    return value

# DATA_ROOT = os.getenv("DATA_ROOT")
# print("DATA_ROOT:", os.getenv("DATA_ROOT"))
# if not DATA_ROOT:
#     raise EnvironmentError(
#         "Could not find DATA_ROOT, make sure the .env file is ok"
#     )

def get_train_data(root_dir: str, download: bool = False) -> datasets.CIFAR10:
    cifar10_dir = os.path.join(root_dir, "cifar-10-batches-py")
    need_download = download or (not os.path.exists(cifar10_dir))
    if need_download:
        print("Downloading CIFAR10 train dataset...")
    else:
        print("CIFAR10 train dataset found, loading...")

    return datasets.CIFAR10(root=root_dir, train=True, download=need_download, transform=None)


def get_test_data(root_dir: str, download: bool = False, transform=None) -> datasets.CIFAR10:
    cifar10_dir = os.path.join(root_dir, "cifar-10-batches-py")
    need_download = download or (not os.path.exists(cifar10_dir))
    if need_download:
        print("Downloading CIFAR10 test dataset...")
    else:
        print("CIFAR10 test dataset found, loading...")

    return datasets.CIFAR10(root=root_dir, train=False, download=need_download, transform=transform)


def get_dataloaders(root_dir: str, batch_size: int = 128, download: bool = False, seed: int = 42):

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])

    # Load train dataset without applying any transforms, doing that in the next step
    dataset = get_train_data(root_dir=root_dir, download=download)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # IMPORTANT: random_split keeps references to the same underlying dataset object.
    # If we set dataset.transform, it affects BOTH splits, and we end up applying training transforms to validation data
    # So we "clone" by creating two separate CIFAR10 objects:
    train_dataset.dataset = datasets.CIFAR10(root=root_dir, train=True, download=False, transform=train_transforms)
    val_dataset.dataset   = datasets.CIFAR10(root=root_dir, train=True, download=False, transform=val_transforms)
    test_dataset = get_test_data(root_dir=root_dir, download=download, transform=val_transforms)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
