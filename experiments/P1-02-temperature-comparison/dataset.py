from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import Config


def get_dataloaders(config: Config = Config()):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((config.mnist_mean,), (config.mnist_std,))
    ])

    train_dataset = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(config.data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

    return train_loader, test_loader
