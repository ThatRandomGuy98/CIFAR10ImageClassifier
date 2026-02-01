import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import os
from dotenv import load_dotenv
load_dotenv()


def get_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"Missing environment variable: {var_name}")
    return value

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


def stratified_split(dataset: datasets.CIFAR10, val_ratio: float = 0.1, seed: int = 22) -> datasets.CIFAR10:
    targets = dataset.targets
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed
    )
    
    train_idx, val_idx = next(splitter.split(X=targets, y=targets))
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return train_dataset, val_dataset


def get_dataloaders(root_dir: str, batch_size: int = 128, download: bool = False, seed: int = 42):

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
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
    # train_size = int(0.9 * len(dataset))
    # val_size = len(dataset) - train_size
    train_dataset, val_dataset = stratified_split(dataset=dataset)

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
