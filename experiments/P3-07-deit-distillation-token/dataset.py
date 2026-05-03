from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import Config


def get_dataloaders(config: Config = Config()):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(config.data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(config.data_dir, train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader
