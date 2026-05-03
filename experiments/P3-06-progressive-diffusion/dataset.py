from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import Config


def get_dataloaders(config: Config = Config()):
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
