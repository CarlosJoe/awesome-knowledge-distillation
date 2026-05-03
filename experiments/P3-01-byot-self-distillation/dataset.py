from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import Config


def get_dataloaders(config: Config = Config()):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.cifar10_mean, config.cifar10_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.cifar10_mean, config.cifar10_std)
    ])

    train_dataset = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(config.data_dir, train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader
