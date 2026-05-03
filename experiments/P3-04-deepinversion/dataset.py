import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


def get_cifar10_trainloader(batch_size, num_workers, data_dir="./data"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    return DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )


def get_cifar10_testloader(batch_size, num_workers, data_dir="./data"):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    return DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )


def get_generated_dataloader(generated_images, generated_labels, batch_size, num_workers=4):
    dataset = TensorDataset(generated_images, generated_labels)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
