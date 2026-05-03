from PIL import Image

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from config import Config

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


class CIFAR100Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


class CombinedDataset(Dataset):
    def __init__(self, data, targets, is_labeled, transform=None):
        self.data = data
        self.targets = targets
        self.is_labeled = is_labeled
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        target = self.targets[idx]
        is_labeled = self.is_labeled[idx]
        if self.transform:
            img = self.transform(img)
        return img, target, is_labeled


def get_standard_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])


def get_strong_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])


def get_plain_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])


def load_and_split_data(config=Config()):
    train_dataset = datasets.CIFAR100(config.data_dir, train=True, download=True)
    test_transform = get_plain_transform()
    test_dataset = datasets.CIFAR100(config.data_dir, train=False, transform=test_transform)

    num_train = len(train_dataset)
    indices = np.arange(num_train)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(num_train * config.labeled_ratio)
    labeled_indices = indices[:split]
    unlabeled_indices = indices[split:]

    labeled_data = train_dataset.data[labeled_indices]
    labeled_targets = np.array(train_dataset.targets)[labeled_indices].tolist()
    unlabeled_data = train_dataset.data[unlabeled_indices]
    unlabeled_targets = np.array(train_dataset.targets)[unlabeled_indices].tolist()

    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

    return labeled_data, labeled_targets, unlabeled_data, unlabeled_targets, test_loader


def get_labeled_loader(labeled_data, labeled_targets, transform, batch_size):
    dataset = CIFAR100Dataset(labeled_data, labeled_targets, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_unlabeled_loader(unlabeled_data, transform, batch_size):
    targets = [0] * len(unlabeled_data)
    dataset = CIFAR100Dataset(unlabeled_data, targets, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_combined_loader(labeled_data, labeled_targets, unlabeled_data, pseudo_labels, transform, batch_size):
    all_data = np.concatenate([labeled_data, unlabeled_data], axis=0)
    all_targets = labeled_targets + pseudo_labels
    is_labeled = [True] * len(labeled_targets) + [False] * len(pseudo_labels)
    dataset = CombinedDataset(all_data, all_targets, is_labeled, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
