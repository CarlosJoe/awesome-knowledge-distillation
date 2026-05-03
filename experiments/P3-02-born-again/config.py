from dataclasses import dataclass, field


@dataclass
class BornAgainConfig:
    dataset_name: str = "CIFAR-10"
    num_classes: int = 10
    input_channels: int = 3
    image_size: int = 32

    num_generations: int = 3
    temperature: float = 4.0
    lambda_kl: float = 0.5

    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4

    seed: int = 42
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"

    generation_accs: list = field(default_factory=list)

    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)
