from dataclasses import dataclass


@dataclass
class Config:
    T: float = 4.0
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    learning_rate: float = 0.1
    epochs: int = 200
    batch_size: int = 128
    test_batch_size: int = 100
    num_classes: int = 10
    data_dir: str = './data'
    cifar10_mean: tuple = (0.4914, 0.4822, 0.4465)
    cifar10_std: tuple = (0.2470, 0.2435, 0.2616)
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 42
